"""Compare compiler acceptance and measure named-function expansion.

Run after `lake build`. An upstream executable must be built from the source
revision recorded in PORTING.md; this script does not download or build it.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import tempfile


def run(args, timeout=30):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timed out"


def tagged_nodes(value):
    if isinstance(value, dict):
        return int("tag" in value) + sum(map(tagged_nodes, value.values()))
    if isinstance(value, list):
        return sum(map(tagged_nodes, value))
    return 0


def compile_case(binary, source, directory, label, checker):
    code, ir, error = run([binary, "futhark", "-f", str(source)])
    if code:
        return {"status": "timeout" if code == 124 else "lowering-error",
                "error": error.strip() or ir.strip()}
    target = directory / f"{label}.fut_soacs"
    target.write_text(ir)
    code, _, error = run(checker + [str(target)])
    status = "ir-timeout" if code == 124 else "invalid-ir" if code else "valid"
    return {"status": status, "ir_bytes": len(ir),
            "error": error.replace(str(directory), "<temporary>").strip()}


def growth(binary, directory, local=False, polymorphic=False):
    rows = []
    for depth in (2, 4, 6, 8, 10):
        declarations = ["(def (fun (f0 (x Int)) (+ x x)))"]
        declarations += [
            f"(def (fun (f{i} (x Int)) (+ (f{i-1} x) (f{i-1} x))))"
            for i in range(1, depth)
        ]
        if polymorphic:
            declarations = ["(def (t-fun (p0 (&t)) (fn ((x &t)) x)))"]
            declarations += [
                f"(def (t-fun (p{i} (&t)) (fn ((x &t)) "
                f"((t-app p{i-1} &t) ((t-app p{i-1} &t) x)))))"
                for i in range(1, depth)
            ]
            source = "\n".join(declarations + [
                f"(entry (main) ((t-app p{depth-1} Int) 1))"
            ])
        elif local:
            bindings = "\n".join(declaration[5:-1] for declaration in declarations)
            source = f"(entry (main) (let ({bindings}) (f{depth-1} 1)))"
        else:
            source = "\n".join(declarations + [f"(entry (main) (f{depth-1} 1))"])
        kind = "poly" if polymorphic else "local" if local else "top"
        path = directory / f"growth-{kind}-{depth}.remora"
        path.write_text(source)
        code, ast, error = run([binary, "dev", "--monomorphize", "--json", "-f", str(path)])
        if code:
            raise RuntimeError(error or ast)
        code, ir, error = run([binary, "futhark", "-f", str(path)])
        if code:
            raise RuntimeError(error or ir)
        rows.append({"depth": depth, "source_bytes": len(source),
                     "ast_nodes": tagged_nodes(json.loads(ast)), "json_bytes": len(ast),
                     "ir_bytes": len(ir)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--upstream-checker", help="checker built against pinned Futhark sources")
    parser.add_argument("--port", default=".lake/build/bin/remora")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"futhark_version": run(["futhark", "--version"])[1].splitlines()[0],
              "upstream": args.upstream, "port": args.port,
              "upstream_checker": args.upstream_checker or "futhark dev", "cases": []}
    upstream_checker = ([args.upstream_checker] if args.upstream_checker
                        else ["futhark", "dev"])
    with tempfile.TemporaryDirectory(prefix="remora-audit-") as temporary:
        directory = Path(temporary)
        for i, path in enumerate(sorted(Path("tests").rglob("*.remora"))):
            if "lean" in path.parts:
                continue  # Port-specific semantics are tested separately.
            upstream = compile_case(args.upstream, path, directory, f"upstream-{i}",
                                    upstream_checker)
            port = compile_case(args.port, path, directory, f"port-{i}", ["futhark", "dev"])
            report["cases"].append({"file": str(path), "upstream": upstream, "port": port})
        report["growth"] = {"upstream": growth(args.upstream, directory),
                            "port": growth(args.port, directory),
                            "upstream_local": growth(args.upstream, directory, local=True),
                            "port_local": growth(args.port, directory, local=True),
                            "upstream_poly": growth(args.upstream, directory, polymorphic=True),
                            "port_poly": growth(args.port, directory, polymorphic=True)}
    counts = Counter((case["upstream"]["status"], case["port"]["status"])
                     for case in report["cases"])
    report["summary"] = [{"upstream": a, "port": b, "count": count}
                         for (a, b), count in sorted(counts.items())]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["growth"], indent=2))


if __name__ == "__main__":
    main()
