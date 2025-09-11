import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load encoders
encoders = joblib.load("models/encoders.pkl")

def load_model(target):
    safe_name = target.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    return joblib.load(f"models/{safe_name}_model.pkl")

# Load dataset to know all columns
data = pd.read_excel("crack_bacteria_strength_dataset_augmented.xlsx")
all_columns = data.columns.tolist()

st.title("🧪 Crack Healing Prediction Model")
st.write("Select inputs and outputs to predict.")

# Let user pick input and output columns
input_features = st.multiselect("Select input features:", all_columns, default=["Initial Crack Width (mm)", "Final Crack Width (mm)", "Healing Efficiency (%)"])
target_features = st.multiselect("Select target features to predict:", [c for c in all_columns if c not in input_features], default=["Species", "Concentration (cells/ml)", "Age"])

# Collect inputs
sample = {}
for col in input_features:
    if data[col].dtype == "object":
        sample[col] = st.selectbox(f"Select {col}:", data[col].unique())
    else:
        sample[col] = st.number_input(f"Enter {col}:", value=float(data[col].mean()))

# Predict when button pressed
if st.button("🔮 Predict"):
    st.subheader("Prediction Results")

    results = {}
    for target in target_features:
        X = pd.DataFrame([sample])
        model = load_model(target)

        # Drop target col if user mistakenly included
        if target in X.columns:
            X = X.drop(columns=[target])

        pred = model.predict(X)

        # Decode categorical
        if target in encoders:
            pred = encoders[target].inverse_transform(pred)

        results[target] = pred[0]
        st.write(f"**{target} →** {pred[0]}")

    # Plot results
    if results:
        st.subheader("📊 Prediction Visualization")
        fig, ax = plt.subplots()
        ax.bar(results.keys(), [float(v) if not isinstance(v, str) else 0 for v in results.values()])
        ax.set_ylabel("Predicted Value (numeric only)")
        ax.set_title("Prediction Results")
        st.pyplot(fig)
