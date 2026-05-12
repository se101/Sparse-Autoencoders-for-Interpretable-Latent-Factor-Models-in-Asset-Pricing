from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "https://github.com/Feng-CityUHK/EquityCharacteristics.git"
DEFAULT_DEST = ROOT / "external" / "EquityCharacteristics"


def main() -> None:
    args = _parse_args()
    args.dest.parent.mkdir(parents=True, exist_ok=True)

    if args.dest.exists():
        if not (args.dest / ".git").exists():
            raise FileExistsError(f"{args.dest} exists but is not a git checkout.")
        print(f"Updating {args.dest}")
        _run(["git", "-C", str(args.dest), "pull", "--ff-only"])
    else:
        print(f"Cloning {args.repo} into {args.dest}")
        _run(["git", "clone", args.repo, str(args.dest)])

    print(f"EquityCharacteristics is ready at: {args.dest}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch the Feng-CityUHK EquityCharacteristics toolkit.",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Repository URL, default {DEFAULT_REPO}.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Checkout destination, default {DEFAULT_DEST}.",
    )
    return parser.parse_args()


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
