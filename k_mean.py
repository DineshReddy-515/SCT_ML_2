import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
points_df = pd.read_csv('Mall_Customers.csv')

# Select features
X = points_df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use the Elbow Method to find the optimal number of clusters
wcss=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

#plot for wcss(within cluster sum of squares) values from elbow method
plt.figure(figsize=(8,11))
plt.grid(True)
plt.plot(range(1,11),wcss,marker='h')
plt.title("Elbow graph")
plt.ylabel("WCSS(within cluster sum of squares)")
plt.xlabel("No:of clusters")
plt.show()


# Apply K-Means clustering with 5 cluster from elbow graph
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids)


# Add cluster labels to the original dataframe (optional)
points_df['Cluster'] = labels

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=labels, cmap='viridis', marker='o', edgecolor='k', alpha=0.75)
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()



