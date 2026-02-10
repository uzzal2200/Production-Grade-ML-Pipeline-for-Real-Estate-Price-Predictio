"""
Simple local prediction script that loads the trained model and makes predictions.
Searches ZenML artifact store for the latest model.
"""

import pandas as pd
from pathlib import Path
import pickle

def load_model():
    """Load the latest trained model from artifact store."""
    try:
        # Search in artifact store
        zenml_dir = Path.home() / ".zenml" / "local_artifact_store"
        if zenml_dir.exists():
            # Find latest sklearn_pipeline artifact
            for model_dir in sorted(zenml_dir.glob("**/sklearn_pipeline"), reverse=True):
                model_file = model_dir / "model"
                if model_file.exists() and model_file.is_file():
                    with open(model_file, "rb") as f:
                        model = pickle.load(f)
                    print(f"✓ Model loaded from artifact store")
                    return model
        
        # Fallback: Search in mlruns
        mlruns_dir = Path.cwd() / "mlruns"
        if mlruns_dir.exists():
            for run_dir in sorted(mlruns_dir.glob("**/artifacts/model"), reverse=True):
                pkl_file = run_dir / "data" / "model.pkl"
                if pkl_file.exists():
                    with open(pkl_file, "rb") as f:
                        model = pickle.load(f)
                    print(f"✓ Model loaded from mlruns")
                    return model
        
        print("❌ No trained model found")
        print("\nMake sure to run training first:")
        print("   python run_pipeline.py")
        return None
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict(model, features_dict):
    """Make a prediction with the loaded model."""
    # Convert dict to DataFrame with same structure as training
    df = pd.DataFrame([features_dict])
    
    # Make prediction
    prediction = model.predict(df)
    return prediction[0]

if __name__ == "__main__":
    print("Loading trained model...\n")
    
    # Load model
    model = load_model()
    if model is None:
        exit(1)
    
    # Sample house data for prediction
    sample_input = {
        "Order": 1,
        "PID": 5286,
        "MS SubClass": 20,
        "Lot Frontage": 80.0,
        "Lot Area": 9600,
        "Overall Qual": 5,
        "Overall Cond": 7,
        "Year Built": 1961,
        "Year Remod/Add": 1961,
        "Mas Vnr Area": 0.0,
        "BsmtFin SF 1": 700.0,
        "BsmtFin SF 2": 0.0,
        "Bsmt Unf SF": 150.0,
        "Total Bsmt SF": 850.0,
        "1st Flr SF": 856,
        "2nd Flr SF": 854,
        "Low Qual Fin SF": 0,
        "Gr Liv Area": 1710.0,
        "Bsmt Full Bath": 1,
        "Bsmt Half Bath": 0,
        "Full Bath": 1,
        "Half Bath": 0,
        "Bedroom AbvGr": 3,
        "Kitchen AbvGr": 1,
        "TotRms AbvGrd": 7,
        "Fireplaces": 2,
        "Garage Yr Blt": 1961,
        "Garage Cars": 2,
        "Garage Area": 500.0,
        "Wood Deck SF": 210.0,
        "Open Porch SF": 0,
        "Enclosed Porch": 0,
        "3Ssn Porch": 0,
        "Screen Porch": 0,
        "Pool Area": 0,
        "Misc Val": 0,
        "Mo Sold": 5,
        "Yr Sold": 2010,
    }
    
    # Make prediction
    print("\n📊 Making prediction...")
    print(f"Input features: {len(sample_input)} properties")
    
    try:
        predicted_price = predict(model, sample_input)
        print(f"\n✓ Predicted House Price: ${predicted_price:,.2f}")
        print(f"\nModel: Linear Regression")
        print(f"Features used: {len(sample_input)}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()

