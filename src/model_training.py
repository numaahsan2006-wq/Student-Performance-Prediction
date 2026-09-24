from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from preprocessing import (
    TARGET_COLUMN,
    build_preprocessor,
    clean_dataset,
    load_dataset,
    split_features_target,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# PERFORMANCE CATEGORY
# ============================================================

def get_performance_category(score: float) -> str:
    """
    Convert a predicted exam score into a performance level.

    Categories:
        High    : 80–100
        Average : 60–79
        Low     : below 60
    """

    if score >= 80:
        return "High"

    if score >= 60:
        return "Average"

    return "Low"


# ============================================================
# MODEL CREATION
# ============================================================

def create_models(preprocessor):

    models = {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        ),

        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=8,
                        min_samples_leaf=5,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    return models


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    # Keep predictions within valid exam-score range
    predictions = predictions.clip(0, 100)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("STUDENT PERFORMANCE ML MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    print(f"\nOriginal dataset shape: {df.shape}")

    # --------------------------------------------------------
    # 2. Clean dataset
    # --------------------------------------------------------

    df = clean_dataset(df)

    # Correct invalid exam scores
    df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(
        lower=0,
        upper=100,
    )

    print(f"Cleaned dataset shape: {df.shape}")

    # --------------------------------------------------------
    # 3. Separate features and target
    # --------------------------------------------------------

    X, y = split_features_target(df)

    print(f"Features: {X.shape}")
    print(f"Target: {y.shape}")

    # --------------------------------------------------------
    # 4. Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"\nTraining records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # --------------------------------------------------------
    # 5. Build preprocessing
    # --------------------------------------------------------

    preprocessor = build_preprocessor(X_train)

    # --------------------------------------------------------
    # 6. Create models
    # --------------------------------------------------------

    models = create_models(preprocessor)

    results = []

    # --------------------------------------------------------
    # 7. Train and evaluate
    # --------------------------------------------------------

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        results.append(
            {
                "Model": name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )

        print(
            f"MAE  : {metrics['MAE']:.3f}"
        )

        print(
            f"RMSE : {metrics['RMSE']:.3f}"
        )

        print(
            f"R²   : {metrics['R2']:.3f}"
        )

        # Save model
        filename = (
            name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        model_path = MODEL_DIR / filename

        joblib.dump(
            model,
            model_path,
        )

        print(
            f"Saved: {model_path}"
        )

    # --------------------------------------------------------
    # 8. Model comparison
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="RMSE"
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 9. Select best model by RMSE
    # --------------------------------------------------------

    best_model_name = results_df.iloc[0]["Model"]

    best_filename = (
        best_model_name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    best_model_path = MODEL_DIR / best_filename

    # Save a copy with a standard name
    best_model = joblib.load(best_model_path)

    joblib.dump(
        best_model,
        MODEL_DIR / "best_model.joblib",
    )

    print(
        f"\nBest model by RMSE: {best_model_name}"
    )

    print(
        f"Saved best model: "
        f"{MODEL_DIR / 'best_model.joblib'}"
    )

    # --------------------------------------------------------
    # 10. Example prediction
    # --------------------------------------------------------

    sample_prediction = best_model.predict(
        X_test.iloc[[0]]
    )[0]

    sample_prediction = float(
        max(0, min(100, sample_prediction))
    )

    category = get_performance_category(
        sample_prediction
    )

    print("\nExample prediction")
    print("-" * 30)
    print(
        f"Predicted Score: "
        f"{sample_prediction:.2f}"
    )
    print(
        f"Performance Level: "
        f"{category}"
    )


if __name__ == "__main__":
    main()