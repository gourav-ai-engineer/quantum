"""Pre-flight checks and run metadata for GPU experiment runs.

    python scripts/preflight.py check [--allow-cpu]
    python scripts/preflight.py meta --out run_meta.json --experiment NAME [--smoke] [--args-json '{...}']

``check`` exits non-zero unless the pinned library versions are installed, CUDA is
available (unless --allow-cpu, smoke tests only) and head_mask actually changes the model.
``meta`` records the commit, date, arguments and environment next to the results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from qfc.preflight import (
    EXPECTED_VERSIONS,
    PreflightError,
    check_cuda,
    check_head_mask_effective,
    check_versions,
    collect_environment,
)


def _git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def cmd_check(args) -> int:
    problems = check_versions()
    if problems:
        for p in problems:
            print(f"PREFLIGHT FAIL: {p}", file=sys.stderr)
        print(f"expected pins: {EXPECTED_VERSIONS}", file=sys.stderr)
        return 1
    try:
        gpu = check_cuda(args.allow_cpu)
        result = check_head_mask_effective()
    except PreflightError as exc:
        print(f"PREFLIGHT FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"versions OK: {EXPECTED_VERSIONS}")
    print(f"device: {gpu}")
    print(
        "head_mask effective: loss(all ones)={loss_all_ones:.6f} "
        "loss(all zeros)={loss_all_zeros:.6f}".format(**result)
    )
    print("PREFLIGHT OK")
    return 0


def cmd_meta(args) -> int:
    meta = {
        "experiment": args.experiment,
        "commit": _git(args.repo, "rev-parse", "HEAD"),
        "commit_short": _git(args.repo, "rev-parse", "--short=12", "HEAD"),
        "dirty": bool(_git(args.repo, "status", "--porcelain", "--untracked-files=no")),
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "smoke": bool(args.smoke),
        "args": json.loads(args.args_json),
        **collect_environment(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check")
    check.add_argument("--allow-cpu", action="store_true", help="smoke tests only")
    check.set_defaults(fn=cmd_check)
    meta = sub.add_parser("meta")
    meta.add_argument("--out", required=True)
    meta.add_argument("--experiment", required=True)
    meta.add_argument("--smoke", action="store_true")
    meta.add_argument("--args-json", default="{}")
    meta.add_argument("--repo", default=".")
    meta.set_defaults(fn=cmd_meta)
    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
