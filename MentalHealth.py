import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os

# ---------------------------------------------------------------------------
# Since the CSV has 1,000,000 rows we sample 20,000 for fast training.
# The model still learns the same relationships.
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "student_mental_health_burnout_1M.csv")

FEATURES = [
    "stress_level",
    "anxiety_score",
    "sleep_hours",
    "study_hours_per_day"
]
TARGET = "burnout_score"

def _load_and_train():
    """Load dataset (sampled) and return a fitted model + metrics."""
    df = pd.read_csv(DATA_PATH, usecols=FEATURES + [TARGET])

    # Use a sample to keep startup fast
    if len(df) > 20000:
        df = df.sample(n=20000, random_state=42)

    df = df.dropna()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2  = round(r2_score(y_test, y_pred), 4)
    mse = round(mean_squared_error(y_test, y_pred), 4)

    return model, r2, mse

# Train once at import time
model, R2_SCORE, MSE_SCORE = _load_and_train()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predictBurnout(stress_level, anxiety_score, sleep_hours, study_hours):
    """Return predicted burnout_score (0-10) rounded to 2 decimals."""
    values = [[
        float(stress_level),
        float(anxiety_score),
        float(sleep_hours),
        float(study_hours)
    ]]
    result = model.predict(values)[0]
    # Clamp to valid range
    result = max(0.0, min(10.0, float(result)))
    return round(result, 2)

def getModelInfo():
    """Return dict with model coefficients and metrics for display."""
    coefs = dict(zip(FEATURES, [round(c, 4) for c in model.coef_]))
    return {
        "intercept": round(float(model.intercept_), 4),
        "coefficients": coefs,
        "r2": R2_SCORE,
        "mse": MSE_SCORE
    }
