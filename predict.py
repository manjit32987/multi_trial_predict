import joblib
import pandas as pd

# Load encoders
encoders = joblib.load("models/encoders.pkl")

def load_model(target):
    safe_name = target.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    return joblib.load(f"models/{safe_name}_model.pkl")

# Example input (user can change this)
sample = {
    "Initial Crack Width (mm)": 0.7,
    "Final Crack Width (mm)": 0.3,
    "Healing Efficiency (%)": 65,
    "Species": "Bacillus subtilis",
    "Concentration (cells/ml)": 1e7,
    "Age": 28
}

# Choose target to predict
target = "Species"   # Change to "Concentration (cells/ml)" or "Age"

# Prepare input
X = pd.DataFrame([sample]).drop(columns=[target])

# Load model
model = load_model(target)

# Predict
y_pred = model.predict(X)

# Decode if categorical
if target in encoders:
    y_pred = encoders[target].inverse_transform(y_pred)

print(f"\n🎯 Predicted {target}: {y_pred[0]}")
