# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print("Program Started")

# Load dataset
data = pd.read_csv("climate_environment_risk_dataset.csv")

# Select numerical features
X = data[['CO2_Level_ppm','Temperature_C','Rainfall_mm',
          'Humidity_percent','Sea_Level_Rise_mm',
          'Deforestation_Rate_percent','Industrial_Emission_Index']]

# ----------------------------
# Elbow Method
# ----------------------------

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.show()

# ----------------------------
# K-Means Clustering
# ----------------------------

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X)

data['Cluster'] = clusters

print("\nCluster Results:")
print(data[['Region','Cluster']].head())

# ----------------------------
# PCA Visualization
# ----------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure()

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters
)

# Plot centroids
centroids = pca.transform(kmeans.cluster_centers_)

plt.scatter(
    centroids[:,0],
    centroids[:,1],
    marker='X',
    s=200
)

plt.title("K-Means Clustering Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()
