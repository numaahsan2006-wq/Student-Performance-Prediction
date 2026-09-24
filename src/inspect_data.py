from pathlib import Path
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = PROJECT_ROOT / "data" / "StudentPerformanceFactors.csv"


def main():
    """Load and inspect the student performance dataset."""

    print("=" * 60)
    print("STUDENT PERFORMANCE DATASET INSPECTION")
    print("=" * 60)

    # Check whether the file exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset path: {DATA_PATH}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n--- COLUMN NAMES ---")
    for column in df.columns:
        print(f"- {column}")

    print("\n--- FIRST 5 ROWS ---")
    print(df.head())

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    print("\n--- DUPLICATE ROWS ---")
    print(df.duplicated().sum())

    print("\n--- BASIC STATISTICS ---")
    print(df.describe(include="all").transpose())


if __name__ == "__main__":
    main()