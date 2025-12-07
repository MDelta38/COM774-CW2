import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the dataset')
    parser.add_argument('--model_type', type=str, default='rf', choices=['rf', 'gb', 'ensemble'])
    parser.add_argument('--run_version', type=str, default='v1')
    args = parser.parse_args()
    
    print(f"Data path: {args.data_path}")
    print(f"Model type: {args.model_type}")
    print(f"Run version: {args.run_version}")
    
    # Load data
    df = pd.read_csv(args.data_path)
    print(f"Original shape: {df.shape}")
    
    # ===== IMPROVEMENT: Feature Engineering =====
    df['COMPLEXITY_DENSITY'] = df['CYCLO'] / (df['LOC'] + 1)
    df['OPERATOR_RATIO'] = df['NUM_OPERATORS'] / (df['NUM_OPERANDS'] + 1)
    df['BRANCH_PER_LINE'] = df['BRANCH_COUNT'] / (df['LOC'] + 1)
    
    # ===== IMPROVEMENT: Remove highly correlated features =====
    correlation_matrix = df.corr().abs()
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
    if to_drop:
        print(f"Dropping correlated features: {to_drop}")
        df = df.drop(columns=to_drop)
    
    print(f"Enhanced shape: {df.shape}")
    
    # Split features/target
    X = df.drop('DEFECT_LABEL', axis=1)
    y = df['DEFECT_LABEL']
    
    print(f"Features: {list(X.columns)}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # ===== IMPROVEMENT: Better train/test split =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # Ensures 50/50 in both sets
    )
    
    print(f"\nTrain class balance: {y_train.value_counts().to_dict()}")
    print(f"Test class balance: {y_test.value_counts().to_dict()}")
    
    # ===== IMPROVEMENT: Start MLflow run with tags =====
    mlflow.start_run(
        run_name=f"defect_model_{args.run_version}",
        tags={
            "model_type": args.model_type,
            "dataset": "enhanced",
            "purpose": "cw2_improvement",
            "features_count": str(X.shape[1]),
            "student": "your_name"
        }
    )
    
    # ===== IMPROVEMENT: Model selection =====
    if args.model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 20)
        
    elif args.model_type == 'gb':
        model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("learning_rate", 0.1)
    
    print(f"\nTraining {args.model_type} model...")
    model.fit(X_train, y_train)
    
    # ===== IMPROVEMENT: Cross-validation =====
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"\n5-Fold Cross-Validation F1 Scores: {cv_scores}")
    print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    mlflow.log_metric("cv_mean_f1", cv_scores.mean())
    mlflow.log_metric("cv_f1_std", cv_scores.std())
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Log metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    mlflow.log_metric("true_negatives", cm[0, 0])
    mlflow.log_metric("false_positives", cm[0, 1])
    mlflow.log_metric("false_negatives", cm[1, 0])
    mlflow.log_metric("true_positives", cm[1, 1])
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        for feature, importance in sorted(feature_importance.items(), 
                                          key=lambda x: x[1], reverse=True)[:5]:
            mlflow.log_metric(f"importance_{feature}", importance)
    
    # Save model
    mlflow.sklearn.log_model(model, "defect_model")
    
    # Print results
    print("\n" + "="*60)
    print(f"MODEL {args.run_version.upper()} - {args.model_type.upper()}")
    print("="*60)
    print(f"\nPerformance Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"CV Mean F1: {cv_scores.mean():.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"TN: {cm[0,0]} | FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")
    
    if hasattr(model, 'feature_importances_'):
        print(f"\nTop 5 Important Features:")
        for feature, importance in sorted(feature_importance.items(), 
                                          key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {feature}: {importance:.4f}")
    
    # ===== IMPROVEMENT: Compare with baseline =====
    if args.run_version != 'baseline':
        print(f"\nImprovement from baseline (61.11%):")
        print(f"Accuracy change: {((accuracy - 0.6111) / 0.6111 * 100):+.1f}%")
    
    mlflow.end_run()
    print(f"\n✅ Run {args.run_version} complete!")
    print("✅ Check Azure ML Studio for comparison!")

if __name__ == "__main__":
    main()