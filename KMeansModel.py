import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ===================== RUTA CORRECTA =====================
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Credit_Card_Dataset.csv")

def _load_and_train():
    print(f"Cargando dataset desde: {DATA_PATH}")  # Para ver si encuentra el archivo
    
    df = pd.read_csv(DATA_PATH)
    
    features = ['Age', 'Annual_Income', 'Credit_Score', 'Credit_Utilization_Ratio',
                'Debt_To_Income_Ratio', 'Number_of_Late_Payments', 'Total_Spend_Last_Year']
    
    X = df[features].copy()
    X = X.fillna(X.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    
    df['cluster'] = kmeans.labels_
    
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids, columns=features).round(2)
    
    # Gráfico
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ['#2ecc71', '#3498db', '#f1c40f', '#e74c3c', '#9b59b6']
    
    for i in range(5):
        cluster_data = df[df['cluster'] == i]
        ax.scatter(cluster_data['Age'], cluster_data['Annual_Income'], 
                  c=colors[i], label=f'Cluster {i}', s=50, alpha=0.85)
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income')
    ax.set_title('Credit Card Customer Segmentation - K-Means')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=180)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return df, centroids_df, plot_base64, scaler, kmeans, features


# Entrenamiento
df_clusters, centroids_df, scatter_plot_img, scaler, model, feature_names = _load_and_train()

cluster_names = {
    0: "Clientes Premium",
    1: "Clientes Jóvenes Activos",
    2: "Clientes Estables",
    3: "Clientes de Alto Riesgo",
    4: "Clientes de Bajo Uso"
}

def get_model_info():
    summary = df_clusters.groupby('cluster').agg({
        'Age': 'mean',
        'Annual_Income': 'mean',
        'Credit_Score': 'mean',
        'Credit_Utilization_Ratio': 'mean',
        'Debt_To_Income_Ratio': 'mean',
        'Number_of_Late_Payments': 'mean',
        'Total_Spend_Last_Year': 'mean',
        'cluster': 'count'
    }).round(2)
    
    summary.rename(columns={'cluster': 'Count'}, inplace=True)
    summary = summary.reset_index()
    summary['Cluster_Name'] = summary['cluster'].map(cluster_names)
    
    return {
        "n_clusters": 5,
        "centroids": centroids_df.to_dict('records'),
        "cluster_summary": summary.to_dict('records'),
        "scatter_plot": scatter_plot_img,
        "features": feature_names,
        "cluster_names": cluster_names
    }

def predict_cluster(age, annual_income, credit_score, utilization, dti, late_payments, spend):
    input_data = pd.DataFrame([{
        'Age': age,
        'Annual_Income': annual_income,
        'Credit_Score': credit_score,
        'Credit_Utilization_Ratio': utilization,
        'Debt_To_Income_Ratio': dti,
        'Number_of_Late_Payments': late_payments,
        'Total_Spend_Last_Year': spend
    }])
    scaled = scaler.transform(input_data)
    cluster_id = int(model.predict(scaled)[0])
    return cluster_id