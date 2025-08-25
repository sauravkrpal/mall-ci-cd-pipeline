import pandas as pd
from sklearn.cluster import KMeans
import joblib
import os

# Load the dataset directly from Mall_Customers.csv
data = pd.read_csv('Mall_Customers.csv')

# Select relevant features for clustering (Income & Spending Score)
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Train a KMeans clustering model
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X)

# Add cluster labels back to the dataset
data['Cluster'] = kmeans.labels_

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('model', exist_ok=True)

# Save processed dataset (with clusters)
data.to_csv('data/clustered_customers.csv', index=False)

# Save the trained model
joblib.dump(kmeans, 'model/customer_segmentation.pkl')

print("Model training completed. Saved at model/customer_segmentation.pkl")
print(f"Total clusters created: {len(set(kmeans.labels_))}")
print("Clustered dataset saved at data/clustered_customers.csv")

# Display cluster summary
print("\nCluster Summary:")
for i in range(5):
    cluster_data = data[data['Cluster'] == i]
    print(f"Cluster {i}: {len(cluster_data)} customers")
    print(f"  - Avg Income: ${cluster_data['Annual Income (k$)'].mean():.1f}k")
    print(f"  - Avg Spending Score: {cluster_data['Spending Score (1-100)'].mean():.1f}")
