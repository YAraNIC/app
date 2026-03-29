import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ecommerce_dataset_updated.csv")

FEATURES    = ['Price (Rs.)', 'Discount (%)', 'Category_enc', 'Payment_enc']
TARGET      = 'premium'

CATEGORIES  = ['Beauty', 'Books', 'Clothing', 'Electronics', 'Home & Kitchen', 'Sports', 'Toys']
PAYMENTS    = ['Cash on Delivery', 'Credit Card', 'Debit Card', 'Net Banking', 'UPI']

# ---------------------------------------------------------------------------
# Train model at import time
# ---------------------------------------------------------------------------
def _load_and_train():
    df = pd.read_csv(DATA_PATH)

    # Target: premium purchase (Final_Price >= 300 Rs.)
    df[TARGET] = (df['Final_Price(Rs.)'] >= 300).astype(int)

    # Encode categoricals with fixed order (same as prediction time)
    le_cat = LabelEncoder()
    le_cat.fit(CATEGORIES)
    le_pay = LabelEncoder()
    le_pay.fit(PAYMENTS)

    df['Category_enc'] = le_cat.transform(df['Category'])
    df['Payment_enc']  = le_pay.transform(df['Payment_Method'])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)

    y_pred   = model.predict(X_test_s)
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    report   = classification_report(y_test, y_pred, output_dict=True)

    return model, scaler, le_cat, le_pay, accuracy, report

model, scaler, le_cat, le_pay, ACCURACY, REPORT = _load_and_train()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predictPremium(price, discount, category, payment_method):
    """
    Returns a dict with:
      - prediction: 0 or 1
      - probability: float (0-100%)
      - label: 'Premium' or 'Standard'
    """
    cat_enc = le_cat.transform([category])[0]
    pay_enc = le_pay.transform([payment_method])[0]

    X = np.array([[float(price), float(discount), cat_enc, pay_enc]])
    X_scaled = scaler.transform(X)

    pred  = int(model.predict(X_scaled)[0])
    proba = float(model.predict_proba(X_scaled)[0][1])

    return {
        "prediction":  pred,
        "probability": round(proba * 100, 1),
        "label":       "Premium" if pred == 1 else "Standard"
    }

def getModelInfo():
    """Returns model metrics and coefficients for display."""
    coefs = dict(zip(
        ['Price', 'Discount (%)', 'Category', 'Payment Method'],
        [round(c, 4) for c in model.coef_[0]]
    ))
    return {
        "accuracy":     round(ACCURACY * 100, 2),
        "intercept":    round(float(model.intercept_[0]), 4),
        "coefficients": coefs,
        "precision_0":  round(REPORT['0']['precision'], 3),
        "recall_0":     round(REPORT['0']['recall'], 3),
        "precision_1":  round(REPORT['1']['precision'], 3),
        "recall_1":     round(REPORT['1']['recall'], 3),
    }

def getOptions():
    """Returns category and payment options for the form."""
    return {
        "categories": CATEGORIES,
        "payments":   PAYMENTS
    }
