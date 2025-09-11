import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np

st.set_page_config(page_title="Crack Healing Prediction", layout="wide")
st.title("🧪 Crack Healing Prediction Model")

# =============================
# Load Encoders
# =============================
encoders_path = "models/encoders.pkl"
if os.path.exists(encoders_path):
    encoders = joblib.load(encoders_path)
else:
    encoders = {}

# =============================
# Helper function to load model
# =============================
def load_model(target):
    safe_name = (
        target.replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )
    path = f"models/{safe_name}_model.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"Model for {target} not found.")
        return None

# =============================
# Load dataset (to know columns)
# =============================
files = ["ceirrus.xlsx", "subtallis.xlsx"]
dfs = [pd.read_excel(f) for f in files if os.path.exists(f)]

if dfs:
    data = pd.concat(dfs, axis=0, ignore_index=True)
    st.success(f"📂 Loaded dataset with {len(data)} rows and {len(data.columns)} columns.")
else:
    st.error("❌ No dataset found. Please place Excel files in the project folder.")
    st.stop()

all_columns = data.columns.tolist()

# Filter only columns that have a trained model for targets
model_files = os.listdir("models")
trained_targets = [f.replace("_model.pkl", "").replace("_", " ") for f in model_files if f.endswith("_model.pkl")]

# =============================
# Streamlit UI
# =============================
st.write("Select which features to use as input and which to predict.")

input_features = st.multiselect(
    "Select input features:",
    all_columns,
    default=[c for c in all_columns if c not in trained_targets][:3],  # default first 3
)

# Only allow predicting columns that actually have models
target_features = st.multiselect(
    "Select target features to predict:",
    [c for c in trained_targets if c not in input_features],
    default=[c for c in trained_targets if c not in input_features][:2],
)

# Collect inputs dynamically
sample = {}
for col in input_features:
    if data[col].dtype == "object":
        # Categorical input
        sample[col] = st.selectbox(f"Select {col}:", sorted(data[col].dropna().unique()))
    elif pd.api.types.is_integer_dtype(data[col]):
        # Integer input
        sample[col] = st.number_input(f"Enter {col}:", value=int(data[col].mean()), step=1)
    else:
        # Float input
        sample[col] = st.number_input(f"Enter {col}:", value=float(data[col].mean()))

# =============================
# Run Prediction
# =============================
if st.button("🔮 Predict"):
    st.subheader("Prediction Results")

    results = {}
    numeric_results = {}

    for target in target_features:
        X = pd.DataFrame([sample])

        # drop target col if mistakenly included
        if target in X.columns:
            X = X.drop(columns=[target])

        model = load_model(target)
        if model is None:
            continue

        # Reindex features to match training
        model_features = model.feature_names_in_
        X = X.reindex(columns=model_features, fill_value=0)

        # Predict
        pred = model.predict(X)

        # Decode categorical
        if target in encoders:
            pred = encoders[target].inverse_transform(pred.astype(int))

        pred_value = pred[0]
        results[target] = pred_value

        # Store numeric separately for plotting
        if isinstance(pred_value, (int, float, np.integer, np.floating)):
            numeric_results[target] = float(pred_value)

        st.write(f"**{target} →** {pred_value}")

    # =============================
    # Visualization
    # =============================
    if numeric_results:
        st.subheader("📊 Numeric Prediction Visualization")
        fig, ax = plt.subplots()
        ax.bar(numeric_results.keys(), numeric_results.values(), color="skyblue")
        ax.set_ylabel("Predicted Value")
        ax.set_title("Numeric Prediction Results")
        st.pyplot(fig)
