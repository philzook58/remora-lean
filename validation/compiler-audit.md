# Compiler audit

The checkpoint is `a9e01ed`. This audit keeps readability and implementation size
as constraints; it does not propose copying every upstream abstraction.

## Acceptance comparison

The Haskell baseline was built from Remora commit
`cb245ea386f8ed8f5ac72957ed51990bebe84bc8` and its pinned Futhark sources at
`16c6e619d1cc1a0fc2497a3da4573a89209ae0b9`. The baseline's required modules were
compiled directly with GHC 9.10.3, `-O0`, and the dependencies selected by Cabal.
No Remora or Futhark source changes were needed. The checker uses the same
parse/alias-analysis/type-check sequence as upstream `futhark dev`.

| Outcome across 106 upstream files | Files |
| --- | ---: |
| Both emit IR accepted by their corresponding checker | 85 |
| Both reject before emitting IR | 21 |
| Rejected only by the Lean port | 0 |

The 21 rejected files break down as follows:

- 11 existential cases: `box0`–`box4`, `iota0`, `mono8`–`mono10`, `reify-shape0`, `sum1`.
- 3 higher-order cases: `import0`, `mono1`, `mono6`.
- 6 negative cases: the five `issue34`/`issue35` shape/type errors and `typecheck0`.
- 1 imported library without an entry point: `dotlib`.

Upstream often crashes for unsupported forms; the port generally returns an error.
The failure mechanisms differ. Matching acceptance on these files does not establish
language equivalence or correctness for untested inputs.

**Backend versions matter.** Upstream emits the newer pinned Futhark IR format;
the port currently emits the format accepted by installed Futhark 0.25.33. Trying
to validate upstream output with that older CLI caused all 85 successful lowerings
to appear invalid. The comparison therefore uses the pinned source checker for
upstream and `futhark dev` 0.25.33 for the port. Unifying that backend target remains
separate work; these results do not claim wire-format compatibility.

## Ordinary named functions: a small fix was warranted

A chain of ten ordinary functions, each calling the previous function twice,
exposed unconditional inlining. The program is only 441 bytes long.

| Ten top-level functions | AST nodes after monomorphization | JSON bytes |
| --- | ---: | ---: |
| Port at checkpoint | 60,369 | 4,975,253 |
| Port after fix | 1,159 | 53,186 |
| Upstream | 1,083 | 169,465 |

AST-node counts demonstrate the structural difference; JSON sizes also include
pretty-printing differences. Increasing depth by two previously multiplied the
port's AST size by roughly four. It now grows linearly, as does emitted IR size.
The equivalent local-function chain now also grows linearly (1,229 AST nodes and
2,740 IR bytes at depth ten).

The fix preserves ordinary function bindings during specialization. Existing
lambda lifting hoists local functions once and substitutes their closure references.
Global function names are excluded from captures. This reuses the current
implementation rather than introducing a specialization cache for monomorphic code.

Regression tests bound AST/IR growth, execute both ten-function chains through C,
and check a lifted lambda calling a global function and local functions capturing
a value. The latter also produces the same result in the upstream interpreter.

## Polymorphic instances: sharing is still worth investigating

A different chain repeatedly instantiates the preceding polymorphic function at
the same type. Its intermediate expansion remains exponential in the port.

| Ten polymorphic definitions | AST nodes | JSON bytes | Final IR bytes |
| --- | ---: | ---: | ---: |
| Port | 13,311 | 1,120,625 | 104 |
| Upstream | 1,053 | 171,646 | 3,907 |

Here the final port IR is tiny: this example demonstrates wasted intermediate
work, not large generated machine code. Upstream's specialization cache prevents
that expansion. A small cache keyed by a definition and its type/index arguments
is the next justified compiler change. It needs tests for captured values and
indices, fresh names, and separation of distinct instances. There is no evidence
from this corpus that a wholesale replacement of all passes is needed for coverage.

## Reproduction

After building the port and upstream, build the audit-only checker against the
pinned upstream Futhark library:

```sh
# Run in the upstream checkout after cabal build:
cabal exec -- ghc -XGHC2021 -XOverloadedStrings -package futhark \
  /path/to/remora-lean/scripts/check_upstream_ir.hs -o /tmp/upstream-ir-check

# Run in the Lean checkout:
python3 scripts/audit_compiler.py --upstream /path/to/upstream/remora \
  --upstream-checker /tmp/upstream-ir-check --output validation/compiler-audit.json
python3 -m unittest discover -s scripts
```

The Haskell checker is validation tooling only; the Lean application does not
invoke it. `compiler-audit.json` contains per-file diagnostics and growth data.
Python growth checks run in CI; C execution is skipped when Futhark or `cc` is absent.
