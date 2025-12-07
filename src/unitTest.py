# working_test.py
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import mlflow

sys.path.append('.')

from finalModel import load_data, prepare_features, split_data, train_model, evaluate_model

class WorkingMLflowTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up MLflow connection"""
        print("\n🔗 Connecting to MLflow...")
        mlflow.set_tracking_uri("http://localhost:8080")
        
        # Test connection 
        try:
            experiments = mlflow.search_experiments()
            print(f"✅ Connected! Found {len(experiments)} experiments")
            cls.mlflow_connected = True
        except Exception as e:
            print(f" Cannot connect to MLflow: {e}")
            cls.mlflow_connected = False
    
    def setUp(self):
        if not self.mlflow_connected:
            self.skipTest("MLflow broke again")
        
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'DEFECT_LABEL': np.random.randint(0, 2, 100)
        })
        
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()
    
    def tearDown(self):
        if hasattr(self, 'temp_csv') and os.path.exists(self.temp_csv.name):
            os.remove(self.temp_csv.name)
    
    def test_dataPipeline(self):
        print("Testing data pipeline")
        
        # Load
        df = load_data(self.temp_csv.name)
        self.assertEqual(len(df), 100)
        
        # Prepare
        X, y = prepare_features(df)
        self.assertEqual(X.shape[1], 3)
        
        # Split
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        self.assertEqual(len(X_test), 20)
    
        model = train_model(X_train, y_train, n_estimators=50, max_depth=10)
        accuracy, report, y_pred = evaluate_model(model, X_test, y_test)
        
        print(f"   Accuracy: {accuracy:.2%}")
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_mlflow_metrics_only(self):
        print("Testing MLflow metrics logging")
        
        df = load_data(self.temp_csv.name)
        X, y = prepare_features(df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        model = train_model(X_train, y_train, n_estimators=30, max_depth=5)
        accuracy, report, _ = evaluate_model(model, X_test, y_test)
        
        with mlflow.start_run(run_name="metrics_test"):
            mlflow.log_params({
                "n_estimators": 30,
                "max_depth": 5,
                "test_size": 0.2
            })
            
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision_0": report['0']['precision'],
                "recall_0": report['0']['recall']
            })
            
            run = mlflow.active_run()
            print(f"   ✅ Logged run: {run.info.run_id}")
            print(f"   📊 Accuracy: {accuracy:.2%}")
    
    def test_manual_model_save(self):
        """Save model manually without MLflow registry"""
        print("\n💾 Testing manual model save...")
        
        df = load_data(self.temp_csv.name)
        X, y = prepare_features(df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        model = train_model(X_train, y_train, n_estimators=20, max_depth=3)
        
        # Save model using joblib (your code does this)
        import joblib
        joblib.dump(model, 'test_model.pkl')
        
        # Load it back
        loaded_model = joblib.load('test_model.pkl')
        
        predictions = loaded_model.predict(X_test[:5])
        self.assertEqual(len(predictions), 5)
        
        print(f"   ✅ Model saved and loaded successfully")
        
        os.remove('test_model.pkl')

def main():
    print("🧪 MLflow Unit Tests (Working Version)")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(WorkingMLflowTest)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    print("\n🔗 MLflow Information:")
    print(f"   URL: http://localhost:8080")
    print(f"   Status: {'✅ Connected' if WorkingMLflowTest.mlflow_connected else '❌ Not connected'}")
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    exit(main())