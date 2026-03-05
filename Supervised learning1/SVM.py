# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

print("Program Started")

# Load dataset
data = pd.read_csv("climate_environment_risk_dataset.csv")

# Features
X = data[['CO2_Level_ppm','Temperature_C','Rainfall_mm',
          'Humidity_percent','Sea_Level_Rise_mm',
          'Deforestation_Rate_percent','Industrial_Emission_Index']]

# Target
y = data['Risk_Level']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SVM model
model = SVC(kernel='linear')

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Visualization of Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot()

plt.title("SVM Confusion Matrix")
plt.show()
