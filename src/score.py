# score.py - SIMPLIFIED VERSION FOR AZURE ML
import json
import joblib
import pandas as pd
import numpy as np
import os

def init():
    global model
    try:
        # Azure ML mounts model here
        model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'defect_model.pkl')
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise

def run(raw_data):
    try:
        # Parse input
        data = json.loads(raw_data)
        
        # Extract features (expecting 20 features)
        if isinstance(data, dict) and 'data' in data:
            features = data['data']
        else:
            features = data
            
        # Convert to DataFrame
        df = pd.DataFrame(features)
        
        # Make prediction
        predictions = model.predict(df)
        
        # Return as JSON
        return json.dumps({
            "predictions": predictions.tolist(),
            "status": "success"
        })
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "status": "failed"
        })