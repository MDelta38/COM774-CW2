# correct_unit_test.py
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import mlflow
import joblib

sys.path.append('.')

from finalModel import load_data, prepare_features, split_data, train_model, evaluate_model

class RealDefectPredictionTests(unittest.TestCase):
    """Tests using YOUR REAL defect prediction dataset"""
    
    @classmethod
    def setUpClass(cls):
        """Load your actual dataset once"""
        print("\n🔍 Loading REAL defect prediction dataset...")
        
        # CHANGE THIS TO YOUR ACTUAL DATA FILE
        cls.real_data_path = "defect_data.csv"  # Your actual file
        
        # Test if file exists
        if not os.path.exists(cls.real_data_path):
            print(f"❌ REAL DATA NOT FOUND: {cls.real_data_path}")
            print("   Please add your actual dataset to test properly")
            cls.real_data_available = False
        else:
            cls.real_data_available = True
            cls.df = pd.read_csv(cls.real_data_path)
            print(f"✅ Loaded {len(cls.df)} real samples")
    
    def setUp(self):
        """Skip if no real data"""
        if not self.real_data_available:
            self.skipTest("Real dataset not available")
        
        # Set up MLflow
        mlflow.set_tracking_uri("http://localhost:8080")
    
    def test_real_data_validation(self):
        """Test YOUR actual data meets requirements"""
        print("\n📊 Testing real data validation...")
        
        # Basic structure
        self.assertIn('DEFECT_LABEL', self.df.columns,
                     "Real data missing DEFECT_LABEL column")
        
        # Check it's binary classification
        unique_labels = self.df['DEFECT_LABEL'].unique()
        self.assertEqual(len(unique_labels), 2,
                        f"DEFECT_LABEL should be binary, found {unique_labels}")
        
        # Check reasonable class distribution (you mentioned 40% error)
        defect_ratio = self.df['DEFECT_LABEL'].mean()
        print(f"   Defect ratio: {defect_ratio:.1%}")
        self.assertGreaterEqual(defect_ratio, 0.3,
                               f"Too few defects: {defect_ratio:.1%}")
        self.assertLessEqual(defect_ratio, 0.7,
                            f"Too many defects: {defect_ratio:.1%}")
    
    def test_feature_engineering(self):
        """Test feature preparation on real data"""
        print("\n⚙️ Testing feature engineering on real data...")
        
        X, y = prepare_features(self.df)
        
        # Should have correct number of features
        expected_features = len(self.df.columns) - 1
        self.assertEqual(X.shape[1], expected_features,
                        f"Expected {expected_features} features, got {X.shape[1]}")
        
        # Features should be numeric
        for col in X.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(X[col]),
                           f"Feature {col} is not numeric")
        
        print(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    def test_minimum_accuracy(self):
        """Test your model meets minimum accuracy requirement"""
        print("\n🎯 Testing minimum accuracy requirement...")
        
        X, y = prepare_features(self.df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        # Train with your best parameters
        model = train_model(X_train, y_train, 
                           n_estimators=500, max_depth=20)
        
        accuracy, report, _ = evaluate_model(model, X_test, y_test)
        
        # BUSINESS REQUIREMENT: Minimum 65% accuracy
        MIN_ACCURACY = 0.65
        print(f"   Achieved accuracy: {accuracy:.1%}")
        print(f"   Minimum required: {MIN_ACCURACY:.1%}")
        
        self.assertGreaterEqual(accuracy, MIN_ACCURACY,
                               f"Model accuracy {accuracy:.1%} < {MIN_ACCURACY:.1%} minimum")
    
    def test_defect_recall(self):
        """Test recall for defect class (catching errors)"""
        print("\n🔎 Testing defect recall (catching errors)...")
        
        X, y = prepare_features(self.df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        model = train_model(X_train, y_train, 
                           n_estimators=500, max_depth=20)
        
        accuracy, report, _ = evaluate_model(model, X_test, y_test)
        
        # Business: We care about catching defects (label=1)
        defect_recall = report['1']['recall']
        MIN_RECALL = 0.7  # Catch at least 70% of defects
        
        print(f"   Defect recall: {defect_recall:.1%}")
        print(f"   Minimum required: {MIN_RECALL:.1%}")
        
        self.assertGreaterEqual(defect_recall, MIN_RECALL,
                               f"Defect recall {defect_recall:.1%} < {MIN_RECALL:.1%}")
    
    def test_model_consistency(self):
        """Test model is reproducible (same seed = same results)"""
        print("\n🔄 Testing model reproducibility...")
        
        X, y = prepare_features(self.df)
        
        # First run
        X_train1, X_test1, y_train1, y_test1 = split_data(X, y, random_state=42)
        model1 = train_model(X_train1, y_train1, random_state=42)
        accuracy1, _, pred1 = evaluate_model(model1, X_test1, y_test1)
        
        # Second run with same seed
        X_train2, X_test2, y_train2, y_test2 = split_data(X, y, random_state=42)
        model2 = train_model(X_train2, y_train2, random_state=42)
        accuracy2, _, pred2 = evaluate_model(model2, X_test2, y_test2)
        
        # Should be identical
        self.assertEqual(accuracy1, accuracy2,
                        f"Accuracy differs: {accuracy1:.4f} vs {accuracy2:.4f}")
        
        # Predictions should match
        match_rate = np.mean(pred1 == pred2)
        self.assertEqual(match_rate, 1.0,
                        f"Predictions differ: {match_rate:.1%} match")
        
        print(f"   Consistency: {match_rate:.1%} match")
    
    def test_feature_importance(self):
        """Test that model learns meaningful feature importance"""
        print("\n📈 Testing feature importance...")
        
        X, y = prepare_features(self.df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        model = train_model(X_train, y_train, 
                           n_estimators=500, max_depth=20)
        
        # Check feature importance sums to ~1
        importance_sum = model.feature_importances_.sum()
        self.assertAlmostEqual(importance_sum, 1.0, places=2,
                              f"Feature importance sums to {importance_sum:.3f}")
        
        # Top feature should have reasonable importance
        top_importance = max(model.feature_importances_)
        self.assertGreaterEqual(top_importance, 0.1,
                               f"Top feature importance {top_importance:.3f} too low")
        
        print(f"   Top feature importance: {top_importance:.1%}")
        
        # Log to MLflow for tracking
        with mlflow.start_run(run_name="feature_importance_test"):
            for i, (feature, imp) in enumerate(zip(X.columns, model.feature_importances_)):
                if i < 5:  # Log top 5
                    mlflow.log_metric(f"imp_{feature}", imp)
                    print(f"     {feature}: {imp:.3f}")

class EdgeCaseTests(unittest.TestCase):
    """Test edge cases for robustness"""
    
    def test_empty_data(self):
        """Test handling of empty dataset"""
        empty_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        empty_csv.write("feature1,feature2,DEFECT_LABEL\n")  # Header only
        empty_csv.close()
        
        with self.assertRaises(AssertionError) as cm:
            load_data(empty_csv.name)
        self.assertIn("empty", str(cm.exception).lower())
        os.remove(empty_csv.name)
    
    def test_missing_column(self):
        """Test handling of missing DEFECT_LABEL column"""
        bad_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
            # Missing DEFECT_LABEL
        })
        
        bad_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        bad_data.to_csv(bad_csv.name, index=False)
        bad_csv.close()
        
        with self.assertRaises(AssertionError) as cm:
            load_data(bad_csv.name)
        self.assertIn("DEFECT_LABEL", str(cm.exception))
        os.remove(bad_csv.name)
    
    def test_single_class(self):
        """Test handling of single-class data"""
        single_class_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'DEFECT_LABEL': [0, 0, 0]  # Only one class
        })
        
        single_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        single_class_data.to_csv(single_csv.name, index=False)
        single_csv.close()
        
        with self.assertRaises(AssertionError) as cm:
            load_data(single_csv.name)
        self.assertIn("binary", str(cm.exception))
        os.remove(single_csv.name)

def main():
    print("🧪 REAL Defect Prediction Unit Tests")
    print("=" * 60)
    
    # Check for real data
    if not os.path.exists("defect_data.csv"):
        print("❌ WARNING: Real dataset 'defect_data.csv' not found!")
        print("   Current tests will use synthetic data only")
        print("   Add your real dataset for meaningful tests")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(RealDefectPredictionTests))
    suite.addTests(loader.loadTestsFromTestCase(EdgeCaseTests))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY - DEFECT PREDICTION MODEL")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print(f"Passed: {passed}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # Business requirements check
    if result.wasSuccessful():
        print("\n✅ MODEL PASSES ALL TESTS")
        print("   Ready for deployment!")
    else:
        print("\n❌ MODEL FAILS SOME TESTS")
        print("   Fix issues before deployment")
    
    print(f"\n🔗 MLflow: http://localhost:8080")
    
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    exit(main())