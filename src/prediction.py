from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    """
    Load the trained best-performing model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# PERFORMANCE CATEGORY
# ============================================================

def get_performance_category(score: float) -> str:
    """
    Convert predicted exam score into a performance category.
    """

    if score >= 80:
        return "High"

    if score >= 60:
        return "Average"

    return "Low"


# ============================================================
# PREDICTION
# ============================================================

def predict_performance(student_data: dict):
    """
    Predict the exam score for a student.

    Parameters
    ----------
    student_data : dict
        Student feature values.

    Returns
    -------
    tuple
        Predicted score and performance category.
    """

    model = load_model()

    # Convert dictionary to one-row DataFrame.
    input_data = pd.DataFrame([student_data])

    # Generate prediction.
    prediction = model.predict(input_data)[0]

    # Keep prediction within valid exam-score range.
    prediction = float(
        max(0, min(100, prediction))
    )

    # Determine performance category.
    category = get_performance_category(
        prediction
    )

    return prediction, category