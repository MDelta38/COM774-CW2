import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import joblib

def load_data(data_path):
    df = pd.read_csv(data_path)
    
    #  validation
    assert 'DEFECT_LABEL' in df.columns, 
    assert len(df) > 0, 
    assert df['DEFECT_LABEL'].nunique() == 2, 
    
    return df

def prepare_features(df):
    """Split features and target"""
    X = df.drop('DEFECT_LABEL', axis=1)
    y = df['DEFECT_LABEL']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    
    return train_test_split( #I decided to use random_state=42 for reproducibilty, the mean accuracy is around 70%
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )

def train_model(X_train, y_train, n_estimators=500, max_depth=20): # n_estimators is the ammount of trees we use, they are like experts in Deepseek which make individual decisions
    # max depth is how deep we go down the decision tree
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report, y_pred

def log_to_mlflow(model, params, metrics, feature_importance):
    """Log everything to MLflow"""
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    
    # Log top features
    for feature, imp in list(feature_importance.items())[:5]:
        mlflow.log_metric(f"imp_{feature}", imp)
    
    mlflow.sklearn.log_model(model, "defect_predictor")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--n_estimators', type=int, default=500, help='Number of trees')
    parser.add_argument('--max_depth', type=int, default=20, help='Max tree depth')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.data_path)
    X, y = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size)
    
    print(f"Training: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Start MLflow run
    mlflow.start_run(run_name="defect_prediction_model")
    
    # Train model
    model = train_model(X_train, y_train, args.n_estimators, args.max_depth)
    
    # Evaluate
    accuracy, report, y_pred = evaluate_model(model, X_test, y_test)
    
    print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Feature importance
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    
    print(f"\nTop 5 Features:")
    for feature, imp in sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feature}: {imp:.4f}")
    
    # Log to MLflow
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "test_size": args.test_size,
        "data_distribution": "50/50_balanced"
    }
    
    metrics = {
        "accuracy": accuracy,
        "precision_0": report['0']['precision'],
        "recall_0": report['0']['recall'],
        "precision_1": report['1']['precision'],
        "recall_1": report['1']['recall']
    }
    
    log_to_mlflow(model, params, metrics, feature_importance)
    
    # Save model locally
    joblib.dump(model, 'defect_model.pkl')
    print(f"\n✅ Model saved as 'defect_model.pkl'")
    
    mlflow.end_run()

if __name__ == "__main__":
    main()

__all__ = [
    'load_data',
    'prepare_features', 
    'split_data',
    'train_model',
    'evaluate_model'
]