#!/usr/bin/env python3
"""Print PyTorch source from Remora's static tensor MLIR.

Requires iree-base-compiler at emission time. The printed module imports only torch.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import re
import shutil
import subprocess
import sys

from iree.compiler import ir


DTYPES = {"f32": "float32", "f64": "float64", "i64": "int64",
          "index": "int64", "i32": "int32", "i1": "bool"}
BINARY = {
    "addi": "+", "addf": "+", "subi": "-", "subf": "-", "muli": "*", "mulf": "*",
    "divf": "/", "andi": "&", "ori": "|", "xori": "^",
}
MATH = {
    "math.sqrt": "sqrt", "math.exp": "exp", "math.log": "log", "math.tanh": "tanh",
    "math.powf": "pow", "math.absf": "abs", "math.sin": "sin", "math.cos": "cos",
    "arith.maximumf": "maximum", "arith.minimumf": "minimum",
}
CASTS = {"arith.index_cast", "arith.sitofp", "arith.uitofp", "arith.extui"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def shape(typ):
    if not isinstance(typ, ir.RankedTensorType):
        return ()
    dims = tuple(ir.RankedTensorType(typ).shape)
    require(all(n >= 0 for n in dims), f"dynamic shape {typ}")
    return dims


def dtype(typ):
    if isinstance(typ, ir.RankedTensorType):
        typ = ir.RankedTensorType(typ).element_type
    name = str(typ)
    require(name in DTYPES, f"unsupported dtype {name}")
    return "torch." + DTYPES[name]


def literal(value):
    if isinstance(value, float) and repr(value) in {"inf", "-inf", "nan"}:
        return f"float({repr(value)!r})"
    return repr(value)


def constant(attr, typ):
    if isinstance(attr, ir.IntegerAttr):
        return literal(ir.IntegerAttr(attr).value), False
    if isinstance(attr, ir.FloatAttr):
        return literal(ir.FloatAttr(attr).value), False
    if isinstance(attr, ir.DenseElementsAttr):
        dense = ir.DenseElementsAttr(attr)
        items = (ir.DenseFPElementsAttr(dense) if dtype(typ).startswith("torch.float")
                 else ir.DenseIntElementsAttr(dense))
        if dense.is_splat:
            return literal(items[0]), True
        return "[" + ", ".join(map(literal, items)) + "]", False
    raise ValueError(f"unsupported constant {attr}")


def generic(op):
    maps = [ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    block = op.regions[0].blocks[0]
    body = [o.operation for o in block.operations]
    require(len(op.results) == 1 and body[-1].name == "linalg.yield",
            "expected a single-result linalg.generic")
    return maps, block.arguments, body[:-1], body[-1].operands[0]


def iterators(op):
    return tuple(str(attr) for attr in op.attributes["iterator_types"])


def projection(mapping):
    if not all(isinstance(a, ir.AffineDimExpr) for a in mapping.results):
        return None
    axes = tuple(a.position for a in mapping.results)
    return axes if len(set(axes)) == len(axes) else None


def float_zero(value):
    """Follow uniform float zero through fills and parallel copies."""
    while isinstance(value, ir.OpResult):
        op = value.owner
        if op.name == "arith.constant":
            attr = op.attributes["value"]
            return isinstance(attr, ir.FloatAttr) and attr.value == 0.0
        if op.name == "linalg.fill":
            value = op.operands[0]
            continue
        if op.name != "linalg.generic":
            return False
        maps, args, body, result = generic(op)
        rank = maps[-1].n_dims
        if (body or iterators(op) != ("#linalg.iterator_type<parallel>",) * rank
                or set(projection(maps[-1]) or ()) != set(range(rank))):
            return False
        if result == args[-1]:
            return False
        value = next((operand for arg, operand in zip(args[:-1], op.operands)
                      if result == arg), result)
    return False


def remove_symmetric_zero_pad(value, axis):
    def pair(v):
        if isinstance(v, ir.OpResult) and v.owner.name == "tensor.concat":
            if ir.IntegerAttr(v.owner.attributes["dim"]).value == axis:
                operands = list(v.owner.operands)
                if len(operands) == 2:
                    return operands
        return None

    outer = pair(value)
    if outer is None:
        return value, 0
    first, second = outer
    if (right := pair(second)) is not None:
        before, middle, after = first, *right
    elif (left := pair(first)) is not None:
        before, middle, after = *left, second
    else:
        return value, 0
    low = shape(before.type)[axis]
    if low != shape(after.type)[axis] or not (float_zero(before) and float_zero(after)):
        return value, 0
    return middle, low


def window_step(expr, spatial, kernel):
    def coefficient(term, axis):
        if isinstance(term, ir.AffineDimExpr) and term.position == axis:
            return 1
        if isinstance(term, ir.AffineMulExpr) and isinstance(term.lhs, ir.AffineDimExpr):
            if term.lhs.position == axis and isinstance(term.rhs, ir.AffineConstantExpr):
                return term.rhs.value
        return 0

    if isinstance(expr, ir.AffineAddExpr):
        for a, b in ((expr.lhs, expr.rhs), (expr.rhs, expr.lhs)):
            stride, dilation = coefficient(a, spatial), coefficient(b, kernel)
            if min(stride, dilation) > 0:
                return stride, dilation
    return None


def strip_broadcast(value, axes):
    """Compose maps through linalg.generic copies without materializing the copies."""
    while isinstance(value, ir.OpResult) and value.owner.name == "linalg.generic":
        op = value.owner
        maps, args, body, result = generic(op)
        incoming = projection(maps[0])
        outgoing = projection(maps[-1])
        if (len(maps) != 2 or body or result != args[0] or incoming is None
                or outgoing != tuple(range(len(axes)))
                or iterators(op) != ("#linalg.iterator_type<parallel>",) * len(axes)):
            break
        axes = tuple(axes[i] for i in incoming)
        value = op.operands[0]
    return value, axes


def contraction(value, axes, out_axes):
    """Recognize a projected multiply feeding a sum over its product."""
    op = value.owner
    if not isinstance(value, ir.OpResult) or op.name != "linalg.generic":
        return None
    maps, args, body, result = generic(op)
    if (len(maps) != 3 or len(body) != 1 or body[0].name != "arith.mulf"
            or result != body[0].results[0] or set(body[0].operands) != set(list(args)[:2])
            or projection(maps[-1]) != tuple(range(len(axes)))
            or iterators(op) != ("#linalg.iterator_type<parallel>",) * len(axes)):
        return None
    parts = []
    for operand, mapping in zip(op.operands, maps[:2]):
        incoming = projection(mapping)
        if incoming is None:
            return None
        parts.append(strip_broadcast(operand, tuple(axes[i] for i in incoming)))
    used = set(parts[0][1]) | set(parts[1][1])
    return parts if used == set(axes) and set(out_axes) <= used else None


class Function:
    def __init__(self, block, names):
        self.names = names
        self.values = {arg: f"v{i}" for i, arg in enumerate(block.arguments)}
        self.params = list(self.values.values())
        self.lines = []
        self.next_id = len(self.params)

    def fresh(self):
        name = f"v{self.next_id}"
        self.next_id += 1
        return name

    def assign(self, expression):
        name = self.fresh()
        self.lines.append(f"{name} = {expression}")
        return name

    def value(self, value):
        if value not in self.values:
            self.values[value] = self.fresh()
        return self.values[value]

    def operation(self, op):
        if not any(result in self.values for result in op.results):
            return
        if op.name == "func.call":
            callee = ir.FlatSymbolRefAttr(op.attributes["callee"]).value
            args = [self.value(value) for value in op.operands] + ["_device=_device"]
            expr = f"{self.names[callee]}({', '.join(args)})"
        else:
            require(len(op.results) == 1, f"{op.name}: expected one result")
            expr = self.expression(op, self.value)
        target = ", ".join(map(self.value, op.results))
        self.lines.append(f"{target} = {expr}")

    def index(self, axis, dims):
        index_shape = tuple(-1 if i == axis else 1 for i in range(len(dims)))
        return f"torch.arange({dims[axis]}, device=_device).reshape({index_shape})"

    def affine(self, expr, dims):
        if isinstance(expr, ir.AffineDimExpr):
            return self.index(expr.position, dims)
        if isinstance(expr, ir.AffineConstantExpr):
            return str(expr.value)
        for cls, symbol in ((ir.AffineAddExpr, "+"), (ir.AffineMulExpr, "*"),
                            (ir.AffineFloorDivExpr, "//"), (ir.AffineModExpr, "%")):
            if isinstance(expr, cls):
                return f"({self.affine(expr.lhs, dims)} {symbol} {self.affine(expr.rhs, dims)})"
        raise ValueError(f"unsupported affine expression {expr}")

    def access(self, name, mapping, dims):
        axes = projection(mapping)
        if axes is None:
            return f"{name}[{', '.join(self.affine(a, dims) for a in mapping.results)}]"
        perm = tuple(axes.index(axis) for axis in sorted(axes))
        if perm != tuple(range(len(axes))):
            name += f".permute({perm})"
        if len(axes) == len(dims):
            return name
        indices = ", ".join(":" if i in axes else "None" for i in range(len(dims)))
        return f"{name}[{indices}]"

    def mapped(self, op):
        maps, args, body, result = generic(op)
        dims = shape(op.results[0].type)
        out_axes = projection(maps[-1])
        rank = maps[-1].n_dims
        require(out_axes is not None, "non-projective output indexing map")
        if rank != len(dims):
            return self.reduce(op, maps, args, body, result, out_axes)

        require(iterators(op) == ("#linalg.iterator_type<parallel>",) * rank,
                "expected parallel iterators")
        iteration_shape = tuple(dims[out_axes.index(i)] for i in range(rank))
        inputs = dict(zip(args[:-1], zip(op.operands, maps[:-1])))
        local = {}

        def get(value):
            require(value != args[-1], "parallel region reads its output initializer")
            if value in inputs:
                operand, mapping = inputs[value]
                return self.access(self.value(operand), mapping, iteration_shape)
            return local[value] if value in local else self.value(value)

        for child in body:
            local[child.results[0]] = self.assign(self.expression(child, get, iteration_shape))
        expr = get(result)
        if out_axes != tuple(range(rank)):
            expr += f".expand({iteration_shape}).permute({out_axes})"
        return f"{expr}.expand({dims})"

    def max_pool(self, op, maps, args, body, result, out_axes):
        if (len(maps) != 3 or out_axes != (0, 1, 2, 3) or projection(maps[1]) != (4, 5)
                or iterators(op) != ("#linalg.iterator_type<parallel>",) * 4
                + ("#linalg.iterator_type<reduction>",) * 2
                or len(body) != 1 or body[0].name != "arith.maximumf"
                or result != body[0].results[0] or set(body[0].operands) != {args[0], args[2]}):
            return None
        coords = list(maps[0].results)
        if coords[:2] != [ir.AffineDimExpr.get(i) for i in range(2)] or len(coords) != 4:
            return None
        y = window_step(coords[2], 2, 4)
        x = window_step(coords[3], 3, 5)
        if y is None or x is None:
            return None
        image, window, init = op.operands
        kernel = shape(window.type)
        strides, dilation = tuple(zip(y, x))
        dims = shape(image.type)
        expected = dims[:2] + tuple((n - d * (k - 1) - 1) // s + 1
                                   for n, d, k, s in zip(dims[2:], dilation, kernel, strides))
        require(expected == shape(op.results[0].type), "unexpected max-pool result shape")
        pool = (f"torch.nn.functional.max_pool2d({self.value(image)}, {kernel}, "
                f"stride={strides}, dilation={dilation})")
        return f"torch.maximum({pool}, {self.value(init)})"

    def reduce(self, op, maps, args, body, result, out_axes):
        pool = self.max_pool(op, maps, args, body, result, out_axes)
        if pool is not None:
            return pool
        require(len(maps) == 2 and len(body) == 1 and result == body[0].results[0]
                and set(body[0].operands) == set(args), "unsupported reduction body")
        kind = {"arith.addf": "sum", "arith.addi": "sum",
                "arith.maximumf": "amax", "arith.minimumf": "amin"}.get(body[0].name)
        axes = projection(maps[0])
        require(kind is not None and axes is not None and set(out_axes) <= set(axes),
                "unsupported reduction indexing or combiner")
        value, init = op.operands
        reduced = tuple(i for i, axis in enumerate(axes) if axis not in out_axes)
        rank = maps[0].n_dims
        expected = tuple("#linalg.iterator_type<parallel>" if i in out_axes
                         else "#linalg.iterator_type<reduction>" for i in range(rank))
        require(len(axes) == rank and reduced and iterators(op) == expected,
                "unsupported reduction axes")
        initial = self.value(init)
        if any(shape(value.type)[i] == 0 for i in reduced):
            return initial
        parts = contraction(value, axes, out_axes) if kind == "sum" else None
        if parts:
            labels = lambda indexes: "".join("abcdefghijklmnopqrstuvwxyz"[i] for i in indexes)
            require(max(axes) < 26, "contraction rank exceeds 26")
            equation = ",".join(labels(idx) for _, idx in parts) + "->" + labels(out_axes)
            tensors = ", ".join(self.value(v) for v, _ in parts)
            expr = f"torch.einsum({equation!r}, {tensors})"
        else:
            cast = f", dtype={dtype(value.type)}" if kind == "sum" else ""
            expr = f"torch.{kind}({self.value(value)}, dim={reduced}{cast})"
            retained = tuple(axis for axis in axes if axis in out_axes)
            if retained != out_axes:
                expr += f".permute({tuple(retained.index(i) for i in out_axes)})"
        if kind == "sum":
            return f"{expr} + {initial}"
        combine = "maximum" if kind == "amax" else "minimum"
        return f"torch.{combine}({expr}, {initial})"

    def expression(self, op, get, dims=None):
        name = op.name
        if name == "linalg.generic":
            return self.mapped(op)
        typ = op.results[0].type
        if name == "arith.constant":
            data, splat = constant(op.attributes["value"], typ)
            layout = "expand" if splat else "reshape"
            return f"torch.tensor({data}, dtype={dtype(typ)}, device=_device).{layout}({shape(typ)})"
        if name == "linalg.index" and dims is not None:
            return self.index(ir.IntegerAttr(op.attributes["dim"]).value, dims)
        if name == "tensor.empty":
            return f"torch.empty({shape(typ)}, dtype={dtype(typ)}, device=_device)"
        if name == "tensor.reshape":
            return f"{get(op.operands[0])}.reshape({shape(typ)})"
        if name == "linalg.conv_2d_nchw_fchw":
            image, weight, init = op.operands
            strides = tuple(ir.DenseIntElementsAttr(op.attributes["strides"]))
            dilation = tuple(ir.DenseIntElementsAttr(op.attributes["dilations"]))
            padding = (0, 0)
            if isinstance(image, ir.OpResult) and image.owner.name == "tensor.reshape":
                source = image.owner.operands[0]
                if shape(image.type) == (1,) + shape(source.type) and len(shape(source.type)) == 3:
                    source, pad_h = remove_symmetric_zero_pad(source, 1)
                    source, pad_w = remove_symmetric_zero_pad(source, 2)
                    padding = (pad_h, pad_w)
            if any(padding):
                image_expr = f"{get(source)}.unsqueeze(0)"
            else:
                image_expr = get(image)
            conv = (f"torch.nn.functional.conv2d({image_expr}, {get(weight)}, "
                    f"stride={strides}, padding={padding}, dilation={dilation})")
            return conv if float_zero(init) else f"{conv} + {get(init)}"
        if name == "linalg.fill":
            return f"{get(op.operands[0])}.expand({shape(typ)})"
        if name == "tensor.concat":
            axis = ir.IntegerAttr(op.attributes["dim"]).value
            nonempty = [v for v in op.operands if shape(v.type)[axis] > 0]
            if len(nonempty) < 2:
                return get(nonempty[0] if nonempty else op.operands[0])
            args = ", ".join(map(get, nonempty))
            return f"torch.cat([{args}], dim={axis})"
        operands = list(map(get, op.operands))
        args = ", ".join(operands)
        if name == "tensor.extract":
            return f"{operands[0]}[{', '.join(operands[1:]) or '()'}]"
        if name == "tensor.insert":
            target = self.assign(f"{operands[1]}.clone()")
            index = ", ".join(operands[2:]) or "()"
            self.lines.append(f"{target}[{index}] = {operands[0]}")
            return target
        if name in {"tensor.insert_slice", "tensor.extract_slice"}:
            offsets, sizes, strides = (
                tuple(ir.DenseI64ArrayAttr(op.attributes[f"static_{key}"]))
                for key in ("offsets", "sizes", "strides")
            )
            require(all(x >= 0 for x in offsets + sizes) and all(x > 0 for x in strides),
                    "slice needs static offsets, sizes, and positive strides")
            index = ", ".join(f"{start}:{start + count * stride}:{stride}"
                              for start, count, stride in zip(offsets, sizes, strides)) or "()"
            if name == "tensor.extract_slice":
                return f"{operands[0]}[{index}].reshape({shape(typ)})"
            target = self.assign(f"{operands[1]}.clone()")
            self.lines.append(f"{target}[{index}] = {operands[0]}.reshape({sizes})")
            return target
        if name == "tensor.from_elements":
            if operands:
                return f"torch.stack([{args}]).reshape({shape(typ)})"
            return f"torch.empty({shape(typ)}, dtype={dtype(typ)}, device=_device)"
        if name in MATH:
            return f"torch.{MATH[name]}({args})"
        if name in CASTS:
            if name in {"arith.uitofp", "arith.extui"}:
                require(dtype(op.operands[0].type) == "torch.bool",
                        "unsigned conversion requires an i1 source")
            return f"{operands[0]}.to({dtype(typ)})"
        if name in {"arith.divsi", "arith.remsi"}:
            return (f"torch.div({args}, rounding_mode='trunc')" if name == "arith.divsi"
                    else f"torch.fmod({args})")
        if name == "arith.select":
            return f"torch.where({args})"
        if name == "arith.cmpf":
            pred = ir.IntegerAttr(op.attributes["predicate"]).value
            symbols = {1: "==", 2: ">", 3: ">=", 4: "<", 5: "<=", 13: "!="}
            require(pred in symbols, "unsupported floating point predicate")
            return f"{operands[0]} {symbols[pred]} {operands[1]}"
        if name == "arith.cmpi":
            pred = ir.IntegerAttr(op.attributes["predicate"]).value
            require(0 <= pred < 6, "unsigned comparisons are unsupported")
            symbol = ("==", "!=", "<", "<=", ">", ">=")[pred]
            return f"{operands[0]} {symbol} {operands[1]}"
        if name.startswith("arith.") and name[6:] in BINARY:
            return f"{operands[0]} {BINARY[name[6:]]} {operands[1]}"
        raise ValueError(f"unsupported operation {name}")

    def render(self, name, operations, result):
        returned = list(map(self.value, result.operands))
        expr = ", ".join(returned) or "()"
        chunks = []
        for op in reversed(operations):
            try:
                self.operation(op)
            except (KeyError, ValueError) as error:
                raise ValueError(f"{op.name}: {error}") from error
            if self.lines:
                chunks.append(self.lines)
                self.lines = []
        lines = [line for chunk in reversed(chunks) for line in chunk] + [f"return {expr}"]
        # Drop eager tensor temporaries after their last use.
        last = {}
        for i, line in enumerate(lines):
            for value in re.findall(r"\bv\d+\b", line):
                last[value] = i
        expired = defaultdict(list)
        for value, i in last.items():
            if i < len(lines) - 1:
                expired[i].append(value)
        params = ", ".join(self.params + ["*", "_device=None"])
        code = [f"def {name}({params}):"]
        if self.params:
            code.append(f"    if _device is None: _device = {self.params[0]}.device")
        for i, line in enumerate(lines):
            code.append(f"    {line}")
            if expired[i]:
                code.append(f"    del {', '.join(expired[i])}")
        return "\n".join(code)


def emit(module):
    functions = list(module.body.operations)
    require(all(op.operation.name == "func.func" for op in functions),
            "expected a module of functions")
    symbols = [ir.StringAttr(op.attributes["sym_name"]).value for op in functions]
    names = {name: f"_fn{i}" for i, name in enumerate(symbols)}
    code = ["import torch"]
    entries = []
    for symbol, fn in zip(symbols, functions):
        if ("sym_visibility" in fn.attributes
                and ir.StringAttr(fn.attributes["sym_visibility"]).value == "private"):
            continue
        entries.append(f"{symbol!r}: {names[symbol]}")
    require(entries, "no public functions")
    for symbol, fn in zip(symbols, functions):
        require(len(fn.regions[0].blocks) == 1, f"{symbol}: expected a single block")
        block = fn.regions[0].blocks[0]
        *operations, result = [op.operation for op in block.operations]
        require(result.name == "func.return", f"{symbol}: missing return")
        code.append(Function(block, names).render(names[symbol], operations, result))
    code.append("entries = {" + ", ".join(entries) + "}")
    return "\n\n\n".join(code) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Remora MLIR file, or - for stdin")
    parser.add_argument("--iree-opt", default=shutil.which("iree-opt")
                        or str(Path(sys.executable).with_name("iree-opt")))
    args = parser.parse_args()
    try:
        source = sys.stdin.read() if args.input == "-" else Path(args.input).read_text()
        optimized = subprocess.run(
            [args.iree_opt, "--inline", "--canonicalize", "--cse", "--symbol-dce"],
            input=source, text=True, stdout=subprocess.PIPE, check=True,
        ).stdout
        with ir.Context():
            print(emit(ir.Module.parse(optimized)), end="")
    except (OSError, ValueError, ir.MLIRError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"PyTorch emitter: {error}\n")


if __name__ == "__main__":
    main()
