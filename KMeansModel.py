import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ===================== LOAD DATASET =====================
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "customer_segmentation_100.csv")

def _load_and_train():
    """Load data, train KMeans and generate visualization"""
    df = pd.read_csv(DATA_PATH)
    
    # Features
    X = df[['age', 'annual_income']]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train K-Means
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    
    # Assign clusters to dataframe
    df['cluster'] = kmeans.labels_
    
    # Get centroids in original scale
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Generate Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for i in range(3):
        cluster_data = df[df['cluster'] == i]
        ax.scatter(cluster_data['age'], cluster_data['annual_income'], 
                  c=colors[i], label=f'Cluster {i+1}', s=60, alpha=0.85)
    
    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], 
              c='yellow', marker='*', s=350, edgecolors='black', 
              linewidth=2.5, label='Centroids')
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income (Rs.)')
    ax.set_title('K-Means Clustering - Customer Segmentation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Convert plot to base64
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=180)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return df, centroids.round(1).tolist(), round(kmeans.inertia_, 2), plot_base64, scaler, kmeans

# Train the model when the file is imported
df_clusters, centroids, wcss, scatter_plot_img, scaler, kmeans_model = _load_and_train()

# ===================== PUBLIC FUNCTIONS =====================

def get_model_info():
    """Return all information needed for the template"""
    summary = df_clusters.groupby('cluster').agg({
        'age': ['count', 'mean'],
        'annual_income': 'mean'
    }).round(1)
    
    summary.columns = ['Count', 'Avg Age', 'Avg Income']
    summary = summary.reset_index()
    
    return {
        "n_clusters": 3,
        "wcss": wcss,
        "centroids": centroids,
        "cluster_summary": summary.to_dict('records'),
        "scatter_plot": scatter_plot_img
    }

def predict_cluster(age, annual_income):
    """Predict which cluster a new customer belongs to"""
    X_new = np.array([[float(age), float(annual_income)]])
    X_scaled = scaler.transform(X_new)
    cluster_id = int(kmeans_model.predict(X_scaled)[0])
    return cluster_id