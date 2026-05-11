from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loading import load_project_data, summarize_project_data  # noqa: E402


def main() -> None:
    data = load_project_data()
    summary = summarize_project_data(data)

    print("Data summary")
    print("------------")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nFactor columns:")
    print(list(data.factors.columns[:10]))
    if len(data.factors.columns) > 10:
        print(f"... ({len(data.factors.columns) - 10} more)")

    print("\nBenchmark columns:")
    print(list(data.benchmarks.columns[:10]))
    if len(data.benchmarks.columns) > 10:
        print(f"... ({len(data.benchmarks.columns) - 10} more)")


if __name__ == "__main__":
    main()
