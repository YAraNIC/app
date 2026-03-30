import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===================== DATASET SINTÉTICO =====================
np.random.seed(42)
n_samples = 100

data = {
    'frequency': np.random.randint(1, 15, n_samples),
    'avg_order': np.random.randint(30, 300, n_samples).astype(float),
    'days_since_last': np.random.randint(1, 60, n_samples),
    'churn': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
}

def _load_and_train():
    df = pd.DataFrame(data)
    X = df[['frequency', 'avg_order', 'days_since_last']]
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    
    model = SGDClassifier(
        loss='log_loss', penalty='l2', alpha=0.001,
        max_iter=2000, tol=1e-3, random_state=42,
        early_stopping=True, validation_fraction=0.1
    )
    model.fit(X_train_s, y_train)
    
    y_pred = model.predict(X_test_s)
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # ==================== MATRIZ DE CONFUSIÓN ====================
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Stay', 'Churn'])
    ax.set_yticklabels(['Stay', 'Churn'])
    
    # Etiquetas dentro de la matriz
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
    
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    confusion_matrix_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return model, scaler, accuracy, report, confusion_matrix_base64

model, scaler, ACCURACY, REPORT, CONFUSION_MATRIX_IMG = _load_and_train()

# ===================== PUBLIC API =====================
def predictChurn(frequency, avg_order, days_since_last):
    X = np.array([[float(frequency), float(avg_order), float(days_since_last)]])
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    proba = float(model.predict_proba(X_scaled)[0][1])
    return {
        "prediction": pred,
        "probability": round(proba * 100, 1),
        "label": "Churn" if pred == 1 else "Stay"
    }

def getModelInfo():
    coefs = dict(zip(['Frequency', 'Avg Order Value', 'Days Since Last'], 
                     [round(c, 4) for c in model.coef_[0]]))
    return {
        "accuracy": round(ACCURACY * 100, 2),
        "intercept": round(float(model.intercept_[0]), 4),
        "coefficients": coefs,
        "precision_1": round(REPORT['1']['precision'], 3),
        "recall_1": round(REPORT['1']['recall'], 3),
        "confusion_matrix_img": CONFUSION_MATRIX_IMG   # ← Nueva clave
    }