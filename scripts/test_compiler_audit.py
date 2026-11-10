"""Run with `python3 -m unittest discover -s scripts` after `lake build`."""
import shutil
import tempfile
from pathlib import Path
import unittest

from audit_compiler import growth, run


class CompilerAuditTests(unittest.TestCase):
    def test_named_function_growth_is_linear(self):
        with tempfile.TemporaryDirectory(prefix="remora-growth-test-") as temporary:
            for local in (False, True):
                with self.subTest(local=local):
                    rows = growth(".lake/build/bin/remora", Path(temporary), local=local)
                    # Five times as many definitions must not cause exponential expansion.
                    self.assertLess(rows[-1]["ast_nodes"], 8 * rows[0]["ast_nodes"])
                    self.assertLess(rows[-1]["ir_bytes"], 8 * rows[0]["ir_bytes"])

    @unittest.skipUnless(shutil.which("futhark") and shutil.which("cc"),
                         "C execution requires Futhark and a C compiler")
    def test_shared_functions_execute_correctly(self):
        with tempfile.TemporaryDirectory(prefix="remora-growth-test-") as temporary:
            directory = Path(temporary)
            for local in (False, True):
                with self.subTest(local=local):
                    growth(".lake/build/bin/remora", directory, local=local)
                    source = directory / f"growth-{'local' if local else 'top'}-10.remora"
                    code, output, error = run([
                        ".lake/build/bin/remora", "futhark", "--run", "-f", str(source),
                    ])
                    self.assertEqual(code, 0, error)
                    self.assertEqual(output.strip(), "1024")


if __name__ == "__main__":
    unittest.main()
