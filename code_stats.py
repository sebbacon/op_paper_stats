"""Analyze OpenPrescribing codebase statistics using cloc."""

import shutil
import subprocess
import sys
from typing import Tuple


def validate_executables() -> Tuple[bool, str]:
    """Check required executables (cloc) are available."""
    if not shutil.which("cloc"):
        return (False, "Missing required executable: cloc")
    return (True, "")


def run_cloc() -> str:
    """Run cloc analysis on Python files and return formatted markdown."""
    result = subprocess.run(
        ["cloc", "--include-lang=Python", "--csv", "."],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"cloc failed: {result.stderr.strip()}")

    # Add debug output
    print("CLOC RAW OUTPUT:\n" + result.stdout, file=sys.stderr)
    
    # Parse CSV output
    for line in result.stdout.split("\n"):
        # Modified parsing logic to handle varying file counts
        if ",Python," in line:
            parts = line.split(",")
            try:
                files = int(parts[0])
                blank = int(parts[2])
                comment = int(parts[3])
                code = int(parts[4])
                total = blank + comment + code
                return (
                    "| Metric | Value |\n"
                    "| --- | --- |\n"
                    f"| Lines of Code | {code:,} |\n"
                    f"| Blank Lines | {blank:,} |\n"
                    f"| Comment Lines | {comment:,} |\n"
                    f"| Total Lines | {total:,} |"
                )
            except (IndexError, ValueError) as e:
                sys.exit(f"Failed to parse cloc output: {e}\nLine: {line}")

    sys.exit(f"Failed to find Python statistics in cloc output. Full output:\n{result.stdout}")


def main() -> None:
    """Main entry point for code statistics analysis."""
    valid, msg = validate_executables()
    if not valid:
        sys.exit(msg)

    print("\n## Code Statistics\n")
    print(run_cloc())


if __name__ == "__main__":
    main()
