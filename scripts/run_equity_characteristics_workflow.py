from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOLKIT_DIR = ROOT / "external" / "EquityCharacteristics"

WORKFLOW_SUBDIR = "char60"
MAIN_SCRIPT = "accounting_60_hxz.py"
MERGE_SCRIPT = "merge_chars_60.py"
IMPUTE_RANK_CANDIDATES = (
    "impute_rank_output_bchmk_60.py",
    "impute_rank_output_bchmk.py",
    "impute_rank_output_bckmk.py",
)


def main() -> None:
    args = _parse_args()
    toolkit_dir = args.toolkit_dir.resolve()
    if not toolkit_dir.exists():
        raise FileNotFoundError(
            f"Missing EquityCharacteristics checkout at {toolkit_dir}. "
            "Run scripts/fetch_equity_characteristics.py first."
        )
    workflow_dir = _workflow_dir(toolkit_dir, args.workflow_subdir)

    python_executable = _resolve_python(args.python)
    commands = _build_commands(workflow_dir, python_executable, args.include_single_characteristics)
    if args.dry_run:
        for command in commands:
            print(" ".join(str(part) for part in command))
        return

    for command in commands:
        print(f"Running: {' '.join(str(part) for part in command)}")
        subprocess.run(command, cwd=workflow_dir, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the documented EquityCharacteristics workflow from a local checkout.",
    )
    parser.add_argument(
        "--toolkit-dir",
        type=Path,
        default=DEFAULT_TOOLKIT_DIR,
        help=f"EquityCharacteristics checkout, default {DEFAULT_TOOLKIT_DIR}.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for the toolkit scripts.",
    )
    parser.add_argument(
        "--workflow-subdir",
        default=WORKFLOW_SUBDIR,
        help=f"Subdirectory containing the characteristic workflow, default {WORKFLOW_SUBDIR}.",
    )
    parser.add_argument(
        "--include-single-characteristics",
        action="store_true",
        help="Also run top-level single-characteristic scripts before merge_chars.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def _workflow_dir(toolkit_dir: Path, workflow_subdir: str) -> Path:
    path = toolkit_dir / workflow_subdir
    if not path.exists():
        raise FileNotFoundError(f"Expected workflow directory {path}")
    return path


def _resolve_python(python_executable: str) -> str:
    path = Path(python_executable)
    if path.exists():
        return str(path.absolute())
    return python_executable


def _build_commands(
    toolkit_dir: Path,
    python_executable: str,
    include_single_characteristics: bool,
) -> list[list[str]]:
    commands: list[list[str]] = []
    commands.append([python_executable, _require_script(toolkit_dir, MAIN_SCRIPT)])

    if include_single_characteristics:
        for script in _discover_single_characteristic_scripts(toolkit_dir):
            commands.append([python_executable, str(script)])

    commands.append([python_executable, _require_script(toolkit_dir, MERGE_SCRIPT)])
    commands.append([python_executable, _find_impute_rank_script(toolkit_dir)])
    return commands


def _require_script(toolkit_dir: Path, script_name: str) -> str:
    path = toolkit_dir / script_name
    if not path.exists():
        raise FileNotFoundError(f"Expected {script_name} in {toolkit_dir}")
    return str(path)


def _find_impute_rank_script(toolkit_dir: Path) -> str:
    for candidate in IMPUTE_RANK_CANDIDATES:
        path = toolkit_dir / candidate
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        f"Could not find an impute/rank output script in {toolkit_dir}. "
        f"Tried: {', '.join(IMPUTE_RANK_CANDIDATES)}"
    )


def _discover_single_characteristic_scripts(toolkit_dir: Path) -> list[Path]:
    excluded = {
        MAIN_SCRIPT,
        MERGE_SCRIPT,
        *IMPUTE_RANK_CANDIDATES,
        "functions.py",
        "iclink.py",
        "pkl_to_csv.py",
    }
    return sorted(
        path
        for path in toolkit_dir.glob("*.py")
        if path.name not in excluded and not path.name.startswith("_")
    )


if __name__ == "__main__":
    main()
