# Credit Risk Assessment Project

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("credit_risk_dataset.csv")

print("Dataset Preview:")
print(data.head())

# Encode target labels
label_encoder = LabelEncoder()
data['Risk'] = label_encoder.fit_transform(data['Risk'])
# Low = 1, High = 0

# Features and target
X = data.drop("Risk", axis=1)
y = data["Risk"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict new customer
new_customer = [[30, 50000, 16000, 620, 4, 1]]

prediction = model.predict(new_customer)

risk_label = label_encoder.inverse_transform(prediction)

print("\nNew Customer Risk Prediction:", risk_label[0])