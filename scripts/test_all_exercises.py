"""Run all weekly exercise tests in order.

Examples:
    python scripts/test_all_exercises.py
    python scripts/test_all_exercises.py --weeks 1 2 13
    python scripts/test_all_exercises.py --fail-fast
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "irlc" / "tests"


@dataclass
class TestResult:
    week: int
    returncode: int
    elapsed: float


def discover_weeks() -> list[int]:
    weeks = []
    for path in TEST_DIR.glob("tests_week*.py"):
        suffix = path.stem.removeprefix("tests_week")
        if suffix.isdigit():
            weeks.append(int(suffix))
    return sorted(weeks)


def run_week(week: int) -> TestResult:
    module = f"irlc.tests.tests_week{week:02d}"
    command = [sys.executable, "-m", module]
    print(f"\n=== Week {week:02d}: {module} ===", flush=True)
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT)
    elapsed = time.perf_counter() - start
    status = "PASS" if completed.returncode == 0 else "FAIL"
    print(f"=== Week {week:02d}: {status} in {elapsed:.1f}s ===", flush=True)
    return TestResult(week=week, returncode=completed.returncode, elapsed=elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all IRLC weekly exercise tests.")
    parser.add_argument(
        "--weeks",
        nargs="+",
        type=int,
        help="Specific week numbers to run. Defaults to every discovered tests_weekNN.py file.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failing week.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    available = discover_weeks()
    weeks = args.weeks if args.weeks is not None else available

    missing = sorted(set(weeks) - set(available))
    if missing:
        available_text = ", ".join(f"{week:02d}" for week in available)
        missing_text = ", ".join(f"{week:02d}" for week in missing)
        print(f"Unknown week(s): {missing_text}. Available weeks: {available_text}", file=sys.stderr)
        return 2

    results: list[TestResult] = []
    total_start = time.perf_counter()
    for week in weeks:
        result = run_week(week)
        results.append(result)
        if args.fail_fast and result.returncode != 0:
            break

    total_elapsed = time.perf_counter() - total_start
    failed = [result for result in results if result.returncode != 0]
    print("\nSummary")
    for result in results:
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"  Week {result.week:02d}: {status} ({result.elapsed:.1f}s)")
    print(f"Total time: {total_elapsed:.1f}s")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
