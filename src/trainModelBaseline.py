import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')
import os

def main():
    # 1. Parse command line arguments - ADD THE MISSING ARGS!
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the dataset')
    parser.add_argument('--model_type', type=str, default='rf', help='Model type (unused in baseline)')
    parser.add_argument('--run_version', type=str, default='baseline', help='Run version tag')
    args = parser.parse_args()
    
    print(f"Data path provided: {args.data_path}")
    print(f"Run version: {args.run_version} (baseline model)")
    
    # 2. Load the dataset from Azure ML
    try:
        df = pd.read_csv(args.data_path)
        print(f"✅ Successfully loaded dataset from Azure ML")
    except Exception as e:
        print(f"❌ Failed to load from {args.data_path}: {e}")
        print("Trying alternative path...")
        # For local testing, you can use a fallback
        df = pd.read_csv('software_defect_data_balanced.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Defect distribution:\n{df['DEFECT_LABEL'].value_counts()}")
    print(f"Columns: {list(df.columns)}")
    
    # 3. Split features/target
    X = df.drop('DEFECT_LABEL', axis=1)
    y = df['DEFECT_LABEL']
    
    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    
    # 5. Start MLflow run (THIS WAS MISSING!)
    mlflow.start_run()
    
    # 6. Train model
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        random_state=42,
        class_weight='balanced'  # Important for 67/33 imbalance
    )
    
    print("Training Random Forest model...")
    model.fit(X_train, y_train)
    
    # 7. Predict and evaluate
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # 8. MLflow logging
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("class_weight", "balanced")
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Log additional metrics
    cm = confusion_matrix(y_test, y_pred)
    mlflow.log_metric("true_negatives", cm[0, 0])
    mlflow.log_metric("false_positives", cm[0, 1])
    mlflow.log_metric("false_negatives", cm[1, 0])
    mlflow.log_metric("true_positives", cm[1, 1])
    
    # 9. Log feature importance
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    for feature, importance in sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True)[:5]:
        mlflow.log_metric(f"importance_{feature}", importance)
    
    # 10. Save model with MLflow
    mlflow.sklearn.log_model(model, "defect_model")
    
    # 11. Print results
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETE")
    print("="*50)
    print(f"\nPerformance Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"TN: {cm[0,0]} | FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")
    
    print(f"\nTop 5 Important Features:")
    for feature, importance in sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feature}: {importance:.4f}")
    
    # 12. End MLflow run
    mlflow.end_run()
    
    print("\n All metrics logged to MLflow!")
    print(" Model saved to MLflow!")
    print(" Check Azure ML Studio for metrics and model registration!")

if __name__ == "__main__":
    main()