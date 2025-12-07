# trainModelRepeat.py
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
from sklearn.utils import resample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the dataset')
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Original dataset: 50% clean, 50% defective")
    print("="*50)
    
    # Split features/target
    X = df.drop('DEFECT_LABEL', axis=1)
    y = df['DEFECT_LABEL']
    
    # Keep test set fixed (20% of data, 50/50 split)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,  # Fixed for consistent test set
        stratify=y
    )
    
    print(f"Test set: {len(X_test)} samples (50% defective)")
    print("="*50)
    
    # Test different defect ratios in TRAINING set only
    defect_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 20% to 80% defective in training
    
    results = []
    
    for target_ratio in defect_ratios:
        print(f"\nTraining with {target_ratio*100:.0f}% defective data in training set:")
        
        # Start MLflow run for this ratio
        mlflow.start_run(run_name=f"train_ratio_{int(target_ratio*100)}")
        mlflow.log_param("target_defect_ratio", target_ratio)
        
        # ===== RESAMPLE TO GET DESIRED RATIO =====
        # Split the remaining 80% into clean/defective
        train_df = pd.concat([X_temp, y_temp], axis=1)
        clean = train_df[train_df['DEFECT_LABEL'] == 0]
        defective = train_df[train_df['DEFECT_LABEL'] == 1]
        
        # Calculate how many of each we need
        total_samples = len(train_df)
        n_defective_needed = int(total_samples * target_ratio)
        n_clean_needed = total_samples - n_defective_needed
        
        # Resample to get desired ratio (with replacement if needed)
        clean_resampled = resample(clean, n_samples=n_clean_needed, random_state=42, replace=True)
        defective_resampled = resample(defective, n_samples=n_defective_needed, random_state=42, replace=True)
        
        # Combine back
        resampled_train = pd.concat([clean_resampled, defective_resampled])
        
        # Shuffle
        resampled_train = resampled_train.sample(frac=1, random_state=42).reset_index(drop=True)
        
        X_train = resampled_train.drop('DEFECT_LABEL', axis=1)
        y_train = resampled_train['DEFECT_LABEL']
        
        
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,  # Fixed for reproducibility
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Predict on FIXED test set (always 50/50)
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
            'target_ratio': target_ratio,
            'accuracy': accuracy,
            'top_feature': top_feature,
            'top_importance': top_importance
        })
        
        print(f"  Accuracy on 50/50 test set: {accuracy:.3f}")
        print(f"  Top feature: {top_feature} ({top_importance:.3f})")
        
        # End this run
        mlflow.end_run()
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY OF DEFECT RATIO EXPERIMENT")
    print("="*50)
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string())
    
    # Find best accuracy
    best_run = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"\n✅ Best accuracy: {best_run['accuracy']:.3f} at {best_run['target_ratio']*100:.0f}% defect ratio")
    print(f"   Top feature was: {best_run['top_feature']}")

if __name__ == "__main__":
    main()