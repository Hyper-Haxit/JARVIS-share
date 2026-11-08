"""Syntax-check the web client's ES modules with Node.

`node --check file.js` does not validate files that use ES-module syntax (it
exits 0), so each module is copied to a temporary `.mjs` file first. A known-bad
control file is checked too, so a broken checker can never report success.

Usage: python scripts/check_web.py   (exit code 0 = all modules valid)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "jarvis" / "clients" / "web" / "static"


def check(path: Path, tmp: Path) -> tuple[bool, str]:
    target = tmp / (path.stem + ".mjs")
    shutil.copyfile(path, target)
    proc = subprocess.run(["node", "--check", str(target)], capture_output=True, text=True)
    return proc.returncode == 0, (proc.stderr or proc.stdout).strip()


def main() -> int:
    if shutil.which("node") is None:
        print("node is not installed; cannot check web modules")
        return 2
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        control = tmp / "control.js"
        control.write_text('import { x } from "./y.js";\nconst a = ;\n', "utf-8")
        ok, _ = check(control, tmp)
        if ok:
            print("checker self-test FAILED: a syntax error was not detected")
            return 2
        failures = 0
        files = sorted(ROOT.rglob("*.js"))
        for f in files:
            ok, out = check(f, tmp)
            print(f"{'ok  ' if ok else 'FAIL'} {f.relative_to(ROOT)}")
            if not ok:
                failures += 1
                print("     " + out.replace("\n", "\n     "))
        print(f"{len(files) - failures}/{len(files)} modules valid")
        # The service worker must cache every module, or an offline start could
        # mix new and old code.
        shell = (ROOT / "sw.js").read_text("utf-8")
        missing = [f for f in files if f.parent != ROOT and f'"/static/{f.relative_to(ROOT).as_posix()}"' not in shell]
        for f in missing:
            print(f"FAIL {f.relative_to(ROOT)} is not in the service worker's cache list (sw.js SHELL)")
        return 1 if failures or missing else 0


if __name__ == "__main__":
    sys.exit(main())
