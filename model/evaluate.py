import pandas as pd
import joblib
from sklearn.metrics import silhouette_score
import numpy as np

# Load the dataset directly from Mall_Customers.csv
data = pd.read_csv('./data/mall.csv')

# Use the same features that were used for training
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Load the trained model
model = joblib.load('model/customer_segmentation.pkl')

# Predict clusters
clusters = model.predict(X)

# Evaluate clustering with silhouette score
score = silhouette_score(X, clusters)

print(f"Silhouette Score of the clustering model: {score:.4f}")
print(f"Number of clusters: {len(set(clusters))}")
print(f"Total customers: {len(data)}")

# Show cluster distribution
unique, counts = np.unique(clusters, return_counts=True)
print("\nCluster Distribution:")
for cluster_id, count in zip(unique, counts):
    percentage = (count / len(data)) * 100
    print(f"Cluster {cluster_id}: {count} customers ({percentage:.1f}%)")

# Show cluster characteristics
print("\nCluster Characteristics:")
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = clusters

for cluster_id in sorted(unique):
    cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster_id]
    print(f"\nCluster {cluster_id}:")
    print(f"  - Average Age: {cluster_data['Age'].mean():.1f}")
    print(f"  - Average Income: ${cluster_data['Annual Income (k$)'].mean():.1f}k")
    print(f"  - Average Spending Score: {cluster_data['Spending Score (1-100)'].mean():.1f}")
    print(f"  - Gender: {dict(cluster_data['Genre'].value_counts())}")
