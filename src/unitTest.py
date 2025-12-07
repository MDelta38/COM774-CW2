# unitTest.py
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

class DefectPredictionTests(unittest.TestCase):
    """Simple unit tests for defect prediction model"""
    
    @classmethod
    def setUpClass(cls):
        """Load dataset once"""
        print("\nLoading defect prediction dataset...")
        cls.data_path = "software_defect_data_balanced.csv"
        
        if not os.path.exists(cls.data_path):
            print(f"Dataset not found: {cls.data_path}")
            cls.data_available = False
        else:
            cls.data_available = True
            cls.df = pd.read_csv(cls.data_path)
            print(f"Loaded {len(cls.df)} samples")
    
    def setUp(self):
        """Skip if no data"""
        if not self.data_available:
            self.skipTest("Dataset not available")
        
        mlflow.set_tracking_uri("http://localhost:8080")
    
    def test1_DataValidation(self):
        """Test 1: Validate dataset structure"""
        print("\nTest 1: Validating dataset structure...")
        
        # Check required column exists
        self.assertIn('DEFECT_LABEL', self.df.columns, 
                     "Missing DEFECT_LABEL column")
        
        # Check binary classification
        unique_labels = self.df['DEFECT_LABEL'].unique()
        self.assertEqual(len(unique_labels), 2,
                        "DEFECT_LABEL should have exactly 2 classes")
        
        print("   PASS: Dataset structure is valid")
    
    def test2_featurePreperation(self):
        """Test 2: Check feature preparation works"""
        print("\nTest 2: Testing feature preparation...")
        
        X, y = prepare_features(self.df)
        
        # Check feature count
        expected_features = len(self.df.columns) - 1
        self.assertEqual(X.shape[1], expected_features,
                        f"Expected {expected_features} features, got {X.shape[1]}")
        
        # Check target matches
        self.assertEqual(len(y), len(self.df))
        
        print(f"   PASS: Prepared {X.shape[1]} features for {len(X)} samples")
    
    def test3_minimumAccuracy(self):
        """Test 3: Model meets minimum accuracy requirement"""
        print("\nTest 3: Testing minimum accuracy...")
        
        X, y = prepare_features(self.df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        # Train model with your best parameters
        model = train_model(X_train, y_train, n_estimators=500, max_depth=20)
        
        accuracy, report, _ = evaluate_model(model, X_test, y_test)
        
        # Minimum 65% accuracy requirement
        MIN_ACCURACY = 0.65
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Required: {MIN_ACCURACY:.1%}")
        
        self.assertGreaterEqual(accuracy, MIN_ACCURACY,
                               f"Accuracy {accuracy:.1%} below {MIN_ACCURACY:.1%} minimum")
        
        print("Success")
    
    def test4_defectDetection(self):
        print("Test 4: Testing defect detection")
        
        X, y = prepare_features(self.df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        model = train_model(X_train, y_train, n_estimators=500, max_depth=20)
        accuracy, report, _ = evaluate_model(model, X_test, y_test)
        
        # Check recall for defect class (label=1)
        defect_recall = report['1']['recall']
        MIN_RECALL = 0.6  # Catch at least 60% of defects
        
        print(f"   Defect recall: {defect_recall:.1%}")
        print(f"   Required: {MIN_RECALL:.1%}")
        
        self.assertGreaterEqual(defect_recall, MIN_RECALL,
                               f"Defect recall {defect_recall:.1%} below {MIN_RECALL:.1%}")
        
        
    
    def test5_MLFlowLogging(self):
        
        print("Test 5: Testing MLflow logging")
        
        # Create a simple test
        with mlflow.start_run(run_name="defect_test"):
            mlflow.log_param("test", "defect_prediction")
            mlflow.log_metric("test_score", 0.85)
            
            # Check run was created
            run = mlflow.active_run()
            self.assertIsNotNone(run)
            
            print(f"   Logged run: {run.info.run_id}")
            print("   PASS: MLflow logging works correctly")

def run_tests():
    print("=" * 60)
    print("Defect Prediction Model - Unit Tests")
    print("=" * 60)
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(DefectPredictionTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    
    print("Result")
    
    print(f"Total tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("SUCCESS: All tests passed")
        print("Model is ready for deployment")
    else:
        print("FAILURE: Some tests failed")
        print("Fix issues before deployment")
    
    print(f"MLflow URL: http://localhost:8080")
    
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    exit(run_tests())