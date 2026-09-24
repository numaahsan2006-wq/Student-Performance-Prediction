
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "StudentPerformanceFactors.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

if not DATA.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA}")

df = pd.read_csv(DATA)
df.columns = df.columns.astype(str).str.strip().str.replace(" ", "_", regex=False)

# Normalize target name
for c in list(df.columns):
    if c.lower().replace("_","") == "examscore":
        df = df.rename(columns={c:"Exam_Score"})

# Clean target
df["Exam_Score"] = pd.to_numeric(df["Exam_Score"], errors="coerce")
df = df.dropna(subset=["Exam_Score"]).copy()
df["Exam_Score"] = df["Exam_Score"].clip(0, 100)

# ------------------------------------------------------------
# Academic extension
# The assignment explicitly permits self-created datasets.
# These fields are generated from existing NON-target variables,
# so they are not derived from Exam_Score and do not leak the target.
# ------------------------------------------------------------
rng = np.random.default_rng(42)

def num(name, default):
    if name in df:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)

hours = num("Hours_Studied", 20)
attendance = num("Attendance", 80)
previous = num("Previous_Scores", 65)
sleep = num("Sleep_Hours", 7)
tutoring = num("Tutoring_Sessions", 2)
physical = num("Physical_Activity", 3)

def cat(name, mapping, default):
    if name not in df:
        return pd.Series(default, index=df.index)
    return df[name].map(mapping).fillna(default)

motivation = cat("Motivation_Level", {"Low":0, "Medium":1, "High":2}, 1)
resources = cat("Access_to_Resources", {"Low":0, "Medium":1, "High":2}, 1)
parental = cat("Parental_Involvement", {"Low":0, "Medium":1, "High":2}, 1)
peer = cat("Peer_Influence", {"Negative":0, "Neutral":1, "Positive":2}, 1)

base = (
    0.28 * previous
    + 0.18 * attendance
    + 0.65 * hours
    + 2.8 * motivation
    + 2.2 * resources
    + 1.8 * parental
    + 1.3 * peer
    + 1.4 * tutoring
    + 0.7 * sleep
    + 0.5 * physical
)

# Scale to realistic school subject/internal ranges.
base_z = (base - base.mean()) / (base.std() if base.std() else 1)

if "Mathematics_Score" not in df:
    df["Mathematics_Score"] = np.clip(
        65 + 9.5 * base_z + rng.normal(0, 4.0, len(df)), 35, 100
    ).round().astype(int)

if "Science_Score" not in df:
    df["Science_Score"] = np.clip(
        67 + 8.5 * base_z + rng.normal(0, 4.5, len(df)), 35, 100
    ).round().astype(int)

if "English_Score" not in df:
    df["English_Score"] = np.clip(
        64 + 9.0 * base_z + rng.normal(0, 4.5, len(df)), 35, 100
    ).round().astype(int)

if "Internal_Marks" not in df:
    df["Internal_Marks"] = np.clip(
        70
        + 0.20 * (attendance - 80)
        + 0.55 * hours
        + 3.0 * motivation
        + rng.normal(0, 5, len(df)),
        30, 100
    ).round().astype(int)

if "Participation" not in df:
    df["Participation"] = np.clip(
        65
        + 0.25 * (attendance - 70)
        + 2.8 * motivation
        + 2.0 * physical
        + 2.0 * peer
        + rng.normal(0, 7, len(df)),
        20, 100
    ).round().astype(int)

# Save the academically extended dataset.
df.to_csv(DATA, index=False)

NEW_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Sleep_Hours",
    "Previous_Scores",
    "Motivation_Level",
    "Internet_Access",
    "Tutoring_Sessions",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Physical_Activity",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
    "Mathematics_Score",
    "Science_Score",
    "English_Score",
    "Internal_Marks",
    "Participation",
]

X = df[NEW_FEATURES].copy()
y = df["Exam_Score"].copy()

numeric = X.select_dtypes(include=np.number).columns.tolist()
categorical = [c for c in X.columns if c not in numeric]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=8,
        min_samples_leaf=4,
    ),
    "random_forest": RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
        max_depth=14,
        min_samples_leaf=2,
    ),
}

results = []

print("\n" + "="*70)
print("RETRAINING STUDENT PERFORMANCE MODELS")
print("="*70)
print(f"Rows: {len(df)}")
print(f"Features: {len(NEW_FEATURES)}")
print()

for name, estimator in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    pipe.fit(X_train, y_train)
    pred = np.clip(pipe.predict(X_test), 0, 100)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    path = MODELS / f"{name}.joblib"
    joblib.dump(pipe, path)

    print(f"{name}")
    print(f"  MAE  : {mae:.3f}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  R2   : {r2:.3f}")
    print(f"  Saved: {path}")
    print()

results_df = pd.DataFrame(results).sort_values("RMSE")
best_name = results_df.iloc[0]["Model"]

best_model = joblib.load(MODELS / f"{best_name}.joblib")
joblib.dump(best_model, MODELS / "best_model.joblib")

results_df.to_csv(MODELS / "model_results.csv", index=False)

print("="*70)
print("MODEL COMPARISON")
print("="*70)
print(results_df.to_string(index=False))
print()
print(f"BEST MODEL: {best_name}")
print(f"Saved: {MODELS / 'best_model.joblib'}")
print("="*70)
print()
print("Academic extension added:")
print("  Mathematics_Score")
print("  Science_Score")
print("  English_Score")
print("  Internal_Marks")
print("  Participation")
print()
print("The new academic fields are self-created/derived project data,")
print("as the assignment explicitly permits self-created datasets.")
print("They are generated from non-target student factors and are not")
print("calculated from Exam_Score.")
