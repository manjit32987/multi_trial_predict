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
# Load and merge datasets
# =============================
files = ["ceirrus.xlsx", "subtallis.xlsx"]
dfs = []

for file in files:
    if os.path.exists(file):
        df = pd.read_excel(file)
        df.columns = df.columns.str.strip().str.title()  # Normalize columns
        df["Source_File"] = os.path.basename(file)
        dfs.append(df)

if dfs:
    data = pd.concat(dfs, axis=0, ignore_index=True)
    st.success(f"📂 Loaded {len(dfs)} files → {len(data)} total rows, {len(data.columns)} columns.")
    st.write("✅ **Preview of combined dataset:**")
    st.dataframe(data.head())
else:
    st.error("❌ No dataset found. Please ensure Excel files are in the same folder as app.py.")
    st.stop()

all_columns = data.columns.tolist()

# =============================
# Detect which columns have models
# =============================
model_files = os.listdir("models")
trained_targets = [
    f.replace("_model.pkl", "").replace("_", " ") 
    for f in model_files if f.endswith("_model.pkl")
]

# =============================
# Streamlit UI for feature selection
# =============================
st.write("### ⚙️ Select Features")

input_features = st.multiselect(
    "Select input features:",
    [c for c in all_columns if c not in trained_targets and c != "Source_File"],
    default=[c for c in all_columns if c not in trained_targets and c != "Source_File"][:3],
)

target_features = st.multiselect(
    "Select target features to predict:",
    [c for c in trained_targets if c not in input_features],
    default=[c for c in trained_targets if c not in input_features][:2],
)

# =============================
# Collect input sample dynamically
# =============================
sample = {}
for col in input_features:
    if data[col].dtype == "object":
        sample[col] = st.selectbox(f"Select {col}:", sorted(data[col].dropna().unique()))
    elif pd.api.types.is_integer_dtype(data[col]):
        sample[col] = st.number_input(f"Enter {col}:", value=int(data[col].mean()), step=1)
    else:
        sample[col] = st.number_input(f"Enter {col}:", value=float(data[col].mean()))

# =============================
# Run Predictions
# =============================
if st.button("🔮 Predict"):
    st.subheader("🎯 Prediction Results")

    results = {}
    numeric_results = {}

    for target in target_features:
        X = pd.DataFrame([sample])

        if target in X.columns:
            X = X.drop(columns=[target])

        model = load_model(target)
        if model is None:
            continue

        # Align columns to training model
        model_features = model.feature_names_in_
        X = X.reindex(columns=model_features, fill_value=0)

        # Predict
        pred = model.predict(X)

        # Decode categorical
        if target in encoders:
            pred = encoders[target].inverse_transform(pred.astype(int))

        pred_value = pred[0]
        results[target] = pred_value

        # Store numeric values for visualization
        if isinstance(pred_value, (int, float, np.integer, np.floating)):
            numeric_results[target] = float(pred_value)

        st.write(f"**{target} →** {pred_value}")

    # =============================
    # Combined Visualization
    # =============================
    st.subheader("📊 Input vs Predicted Visualization")

    input_numeric = {
        k: v for k, v in sample.items()
        if isinstance(v, (int, float, np.integer, np.floating))
    }
    combined_data = {**input_numeric, **numeric_results}

    if combined_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [
            "lightgreen" if k in input_numeric else "skyblue"
            for k in combined_data.keys()
        ]
        bars = ax.bar(combined_data.keys(), combined_data.values(), color=colors)
        ax.set_ylabel("Value")
        ax.set_title("Input Features (Green) vs Predicted Outputs (Blue)")

        # Value labels
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2, yval,
                f"{yval:.2f}", ha='center', va='bottom', fontsize=9
            )

        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.info("No numeric data available for visualization.")
