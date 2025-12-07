# trainModelTune.py
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the dataset')
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Original dataset: 50% clean, 50% defective")
    print("="*60)
    
    # Split features/target
    X = df.drop('DEFECT_LABEL', axis=1)
    y = df['DEFECT_LABEL']
    
    # Standard 80/20 split with 50/50 balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # Keep 50/50 in both sets
    )
    
    print(f"Training: {len(X_train)} samples (50% defective)")
    print(f"Test: {len(X_test)} samples (50% defective)")
    print("="*60)
    
    # Test different hyperparameters
    results = []
    
    # Try increasing depth and number of trees
    n_estimators_list = [50, 100, 200, 300, 500]  # More trees
    max_depth_list = [5, 10, 15, 20, None]  # None = unlimited depth
    
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            run_name = f"RF_{n_estimators}trees_{max_depth}depth"
            print(f"\nTraining: {n_estimators} trees, max_depth={max_depth}")
            
            # Start MLflow run
            mlflow.start_run(run_name=run_name)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth if max_depth else "unlimited")
            
            # Train model with these parameters
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
            
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log accuracy
            mlflow.log_metric("accuracy", accuracy)
            
            # Get top feature
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            top_feature = max(feature_importance, key=feature_importance.get)
            top_importance = feature_importance[top_feature]
            
            mlflow.log_metric("top_feature_importance", top_importance)
            mlflow.log_param("top_feature", top_feature)
            
            # Save results
            results.append({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'accuracy': accuracy,
                'top_feature': top_feature,
                'top_importance': top_importance
            })
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Top feature: {top_feature} ({top_importance:.4f})")
            
            # End this run
            mlflow.end_run()
    
    # Print summary
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    
    # Sort by accuracy
    results_df = results_df.sort_values('accuracy', ascending=False)
    print("\nTop 5 configurations:")
    print(results_df.head(5).to_string())
    
    # Best configuration
    best = results_df.iloc[0]
    print(f"\n✅ Best configuration:")
    print(f"   {best['n_estimators']} trees, max_depth={best['max_depth']}")
    print(f"   Accuracy: {best['accuracy']:.4f}")
    print(f"   Top feature: {best['top_feature']} ({best['top_importance']:.4f})")
    
    # Analyze patterns
    print("\n" + "="*60)
    print("PATTERNS OBSERVED:")
    print("="*60)
    
    # Group by parameter
    by_trees = results_df.groupby('n_estimators')['accuracy'].mean()
    by_depth = results_df.groupby('max_depth')['accuracy'].mean()
    
    print("\nAverage accuracy by number of trees:")
    for n_trees, avg_acc in by_trees.items():
        print(f"  {n_trees:3d} trees: {avg_acc:.4f}")
    
    print("\nAverage accuracy by max depth:")
    for depth, avg_acc in by_depth.items():
        depth_str = "unlimited" if pd.isna(depth) else str(depth)
        print(f"  {depth_str:>9}: {avg_acc:.4f}")

if __name__ == "__main__":
    main()