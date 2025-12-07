# run_local.py
# This script runs the training LOCALLY first to test everything
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import os

print("=== Running Defect Prediction Model Locally ===")

# Load your dataset (make sure it's in the same folder)
data_path = "software_defect_data_balanced.csv"  # Your cleaned dataset from CW1
print(f"Loading data from: {data_path}")

if not os.path.exists(data_path):
    print(f"ERROR: File '{data_path}' not found!")
    print("Please make sure your dataset is in the same folder as this script.")
    exit(1)

df = pd.read_csv(data_path)
print(f"Dataset loaded. Shape: {df.shape}")
print(f"Defect distribution:\n{df['DEFECT_LABEL'].value_counts()}")
print(f"Clean: {df['DEFECT_LABEL'].value_counts()[0]} ({df['DEFECT_LABEL'].value_counts()[0]/len(df)*100:.1f}%)")
print(f"Defective: {df['DEFECT_LABEL'].value_counts()[1]} ({df['DEFECT_LABEL'].value_counts()[1]/len(df)*100:.1f}%)")

# Split features and target
X = df.drop('DEFECT_LABEL', axis=1)
y = df['DEFECT_LABEL']

print(f"\nFeatures used: {list(X.columns)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")

# Train model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== Results ===")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f} (When we predict defect, how often are we right?)")
print(f"Recall:    {recall:.3f} (What percentage of actual defects did we catch?)")
print(f"F1 Score:  {f1:.3f} (Balance between precision and recall)")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Top 5 Most Important Features ===")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# Save model locally
print("\nSaving model locally...")
joblib.dump(model, 'defect_model_local.pkl')
print("Model saved as 'defect_model_local.pkl'")

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("Feature importance saved as 'feature_importance.csv'")

print("\n=== LOCAL TRAINING COMPLETE ===")
print("Next steps:")
print("1. Run the Azure submission script")
print("2. Or use the Azure ML website (ml.azure.com)")