# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
data = pd.read_csv("climate_environment_risk_dataset.csv")

# Select numerical features
X = data[['CO2_Level_ppm',
          'Temperature_C',
          'Rainfall_mm',
          'Humidity_percent',
          'Sea_Level_Rise_mm',
          'Deforestation_Rate_percent',
          'Industrial_Emission_Index']]

# -----------------------------
# Step 1 : Mean
# -----------------------------
mean_values = np.mean(X, axis=0)

print("\nMean of each feature:\n")
print(mean_values)

# -----------------------------
# Step 2 : Standardization
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Step 3 : Covariance Matrix
# -----------------------------
cov_matrix = np.cov(X_scaled.T)

print("\nCovariance Matrix:\n")
print(cov_matrix)

# -----------------------------
# Step 4 : Eigenvalues & Eigenvectors
# -----------------------------
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

print("\nEigen Values:\n")
print(eigen_values)

print("\nEigen Vectors:\n")
print(eigen_vectors)

# -----------------------------
# Step 5 : PCA Transformation
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Convert to dataframe
pca_df = pd.DataFrame(X_pca, columns=["PC1","PC2"])

# -----------------------------
# PCA Visualization
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(pca_df["PC1"], pca_df["PC2"])

plt.title("PCA Visualization of Climate Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid()
plt.show()
