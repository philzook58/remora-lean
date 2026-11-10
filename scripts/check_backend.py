"""Inventory SOACS validity; reports failures without silently excluding fixtures."""
import collections
import json
from pathlib import Path
import subprocess
import tempfile

results = []
with tempfile.TemporaryDirectory(prefix="remora-backend-") as directory:
    for source in sorted(Path("tests").rglob("*.remora")):
        lower = subprocess.run(
            [".lake/build/bin/remora", "futhark", "-f", str(source)],
            capture_output=True, text=True, timeout=30,
        )
        if lower.returncode:
            results.append((str(source), "lowering-error", lower.stderr.strip()))
            continue
        target = Path(directory) / (source.stem + ".fut_soacs")
        target.write_text(lower.stdout)
        checked = subprocess.run(
            ["futhark", "dev", str(target)], capture_output=True, text=True, timeout=30,
        )
        results.append((str(source), "valid" if checked.returncode == 0 else "invalid-ir",
                        checked.stderr.strip()))
print(json.dumps(results, indent=2))
print(dict(collections.Counter(result[1] for result in results)))
