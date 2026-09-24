from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "StudentPerformanceFactors.csv"


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "Exam_Score"


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the student performance dataset.

    Parameters
    ----------
    path : Path
        Path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    df = pd.read_csv(path)

    return df


# ============================================================
# BASIC CLEANING
# ============================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic dataset cleaning.

    Operations:
    - Remove duplicate rows.
    - Validate target column.
    - Normalize column names.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """

    df = df.copy()

    # Remove accidental whitespace from column names
    df.columns = df.columns.str.strip()

    # Remove duplicate records
    df = df.drop_duplicates().reset_index(drop=True)

    # Ensure target exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    return df


# ============================================================
# FEATURE / TARGET SPLIT
# ============================================================

def split_features_target(
    df: pd.DataFrame,
):
    """
    Separate predictor variables from the target.

    Returns
    -------
    X : pd.DataFrame
        Input features.
    y : pd.Series
        Target variable.
    """

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


# ============================================================
# IDENTIFY COLUMN TYPES
# ============================================================

def get_feature_types(X: pd.DataFrame):
    """
    Identify numerical and categorical features.
    """

    numerical_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    return numerical_features, categorical_features


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing pipeline.

    Numerical features:
        Missing values → median
        Scaling → StandardScaler

    Categorical features:
        Missing values → most frequent
        Encoding → OneHotEncoder
    """

    numerical_features, categorical_features = get_feature_types(X)

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor