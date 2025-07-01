"""Analyze OpenPrescribing codebase statistics using cloc."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple


def validate_executables() -> Tuple[bool, str]:
    """Check required executables (git, cloc) are available."""
    missing = []
    for cmd in ["git", "cloc"]:
        try:
            subprocess.run(
                [cmd, "--version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(cmd)

    if missing:
        return (False, f"Missing required executables: {', '.join(missing)}")
    return (True, "")


def clone_repository(output_dir: Path, overwrite: bool) -> None:
    """Perform shallow clone of OpenPrescribing repo."""
    if output_dir.exists():
        if overwrite:
            print(f"Removing existing directory {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(
                f"Output directory {output_dir} already exists\n"
                "Use --overwrite to replace existing clone"
            )

    else:
        clone_cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/bennettoxford/openprescribing",
            str(output_dir),
        ]

        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"Clone failed: {result.stderr.strip()}")


def run_cloc(repo_path: Path) -> str:
    """Run cloc analysis on Python files in repository and return results."""
    result = subprocess.run(
        ["cloc", "--include-lang=Python", "--csv", "."],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        sys.exit(f"cloc failed: {result.stderr.strip()}")

    return result.stdout


def main() -> None:
    """Main entry point for code statistics analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze OpenPrescribing codebase statistics"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("openprescribing-repo"),
        help="Directory to clone repository into (default: openprescribing-repo)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing directory if it exists",
    )

    args = parser.parse_args()

    # Validate environment first
    valid, msg = validate_executables()
    if not valid:
        sys.exit(msg)

    # Clone repository with overwrite flag
    clone_repository(args.output_dir, args.overwrite)

    # Run analysis and print results
    print(f"\nCode statistics for OpenPrescribing:\n")
    print(run_cloc(args.output_dir))


if __name__ == "__main__":
    main()
