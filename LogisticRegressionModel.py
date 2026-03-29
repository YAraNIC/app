import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "ecommerce_dataset_updated.csv")
FEATURES   = ['Price (Rs.)', 'Category_enc', 'Payment_enc', 'month', 'is_weekend']
TARGET     = 'high_savings'
CATEGORIES = ['Beauty', 'Books', 'Clothing', 'Electronics', 'Home & Kitchen', 'Sports', 'Toys']
PAYMENTS   = ['Cash on Delivery', 'Credit Card', 'Debit Card', 'Net Banking', 'UPI']
MONTHS     = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',
              7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}

def _load_and_train():
    df = pd.read_csv(DATA_PATH)
    df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], dayfirst=True)
    df['month']      = df['Purchase_Date'].dt.month
    df['is_weekend'] = df['Purchase_Date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['savings']    = df['Price (Rs.)'] - df['Final_Price(Rs.)']
    df[TARGET]       = (df['savings'] >= 60).astype(int)

    le_cat = LabelEncoder(); le_cat.fit(CATEGORIES)
    le_pay = LabelEncoder(); le_pay.fit(PAYMENTS)
    df['Category_enc'] = le_cat.transform(df['Category'])
    df['Payment_enc']  = le_pay.transform(df['Payment_Method'])

    X = df[FEATURES]; y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

def predictSavings(price, category, payment, month, is_weekend):
    cat_enc = le_cat.transform([category])[0]
    pay_enc = le_pay.transform([payment])[0]
    X = pd.DataFrame([{'Price (Rs.)': float(price), 'Category_enc': cat_enc,
                        'Payment_enc': pay_enc, 'month': int(month), 'is_weekend': int(is_weekend)}])
    X_scaled = scaler.transform(X)
    pred  = int(model.predict(X_scaled)[0])
    proba = float(model.predict_proba(X_scaled)[0][1])
    return {"prediction": pred, "probability": round(proba * 100, 1),
            "label": "High Savings" if pred == 1 else "Low Savings"}

def getModelInfo():
    names = ['Price (Rs.)', 'Category', 'Payment Method', 'Month', 'Is Weekend']
    coefs = dict(zip(names, [round(c, 4) for c in model.coef_[0]]))
    return {"accuracy": round(ACCURACY * 100, 2), "intercept": round(float(model.intercept_[0]), 4),
            "coefficients": coefs,
            "precision_0": round(REPORT['0']['precision'], 3), "recall_0": round(REPORT['0']['recall'], 3),
            "precision_1": round(REPORT['1']['precision'], 3), "recall_1": round(REPORT['1']['recall'], 3),
            "f1_0": round(REPORT['0']['f1-score'], 3),         "f1_1": round(REPORT['1']['f1-score'], 3)}

def getOptions():
    return {"categories": CATEGORIES, "payments": PAYMENTS, "months": MONTHS}
