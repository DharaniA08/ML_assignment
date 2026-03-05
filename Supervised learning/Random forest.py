# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("climate_environment_risk_dataset.csv")

# Features
X = data[['CO2_Level_ppm','Temperature_C','Rainfall_mm',
          'Humidity_percent','Sea_Level_Rise_mm',
          'Deforestation_Rate_percent','Industrial_Emission_Index']]

# Target
y = data['Risk_Level']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=20,
    max_depth=4,        # limits tree size for clean visualization
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------------
# Random Forest Tree Visualization
# -------------------------------

plt.figure(figsize=(22,12))

plot_tree(
    model.estimators_[0],        # visualize first tree
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Random Forest - Tree 1", fontsize=18)
plt.show()
