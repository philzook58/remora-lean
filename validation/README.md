# Local validation

Lean 4.35.0, Z3 4.13.4, Futhark 0.25.33, x86_64 Linux.

`interpreter.txt` records `lake test`: 99 upstream runs plus seven Lean-specific regressions.

`c-supported.txt` records:

```sh
lake exe remora test --modes compile --exclude-tags higher-order,existential,native tests
```

The excluded tags are explicit. The higher-order/existential cases are not certified
as passing and do not count toward the 89 successful C executions. The native tag
contains numeric behavior checks intended for the Lean interpreter. Fifteen upstream
files without directive blocks are retained as fixtures/negative examples.

Additional focused compiled validation passed six runs from basic3, head3, idxs-of,
rec7, and reify-dim0. These exercise frame extension, nested array literals,
function result aliases, record field replication, and static dimension reification.

`lake build` passed. Program AST dumps emitted all six stage JSON files. The full
C corpus and CUDA runtime are not claimed to pass. See ../PORTING.md.

The exact-baseline comparison and code-growth measurements are in
[compiler-audit.md](compiler-audit.md), with per-file results in
[compiler-audit.json](compiler-audit.json). The growth tests run with
`python3 -m unittest discover -s scripts`; C execution requires Futhark and `cc`.
