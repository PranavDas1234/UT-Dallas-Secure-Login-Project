"""Predict one login using the balanced model."""
import joblib
import pandas as pd
from rba_utils import MODEL_DIR, apply_category_maps, basic_clean

MODEL_PATH = MODEL_DIR / "login_threat_model_balanced.joblib"

def predict_login(login_event: dict):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained balanced model found at {MODEL_PATH}. Run train_model_balanced.py first.")
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    target = bundle["target"]
    threshold = bundle["threshold"]
    category_maps = bundle["category_maps"]
    feature_columns = bundle["feature_columns"]
    df = basic_clean(pd.DataFrame([login_event]))
    for col in feature_columns:
        if col not in df.columns:
            df[col] = None
    X = apply_category_maps(df[feature_columns].copy(), category_maps)
    probability = pipeline.predict_proba(X)[0][1]
    prediction = int(probability >= threshold)
    return {
        "target": target,
        "threshold": threshold,
        "prediction": prediction,
        "risk_probability": probability,
        "label": "Suspicious / Malicious" if prediction == 1 else "Normal / Low Risk",
    }

def main():
    example_login = {
        "Country": "US",
        "Region": "New York",
        "City": "Rochester",
        "ASN": 12345,
        "OS Name and Version": "Windows 10",
        "Browser Name and Version": "Chrome 120",
        "Device Type": "desktop",
        "Login Timestamp": "2021-02-01 02:15:00",
        "Round-Trip Time [ms]": 650,
        "Login Successful": True,
        "IP Address": "203.0.113.10",
        "User Agent String": "Mozilla/5.0",
        "User ID": 999999,
        "Is Attack IP": False,
        "Is Account Takeover": False,
    }
    result = predict_login(example_login)
    print("Prediction result")
    print("-" * 40)
    print(f"Target: {result['target']}")
    print(f"Label: {result['label']}")
    print(f"Risk probability: {result['risk_probability']:.4f}")
    print(f"Decision threshold: {result['threshold']:.4f}")

if __name__ == "__main__":
    main()
