import pandas as pd
import joblib
import os

# ========== Step 1: Load new data from two files ==========
file1 = "ceirrus.xlsx"
file2 = "subtallis.xlsx"

df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# Align columns
if not df1.columns.equals(df2.columns):
    print("⚠️ Columns differ. Aligning columns before merging...")
    all_cols = sorted(set(df1.columns).union(set(df2.columns)))
    df1 = df1.reindex(columns=all_cols)
    df2 = df2.reindex(columns=all_cols)

# Merge datasets
df_new = pd.concat([df1, df2], ignore_index=True)
print(f"📊 New data shape: {df_new.shape}")

# ========== Step 2: Load models and encoders ==========
models_folder = "models"
model_files = [f for f in os.listdir(models_folder) if f.endswith("_model.pkl")]
encoders_file = os.path.join(models_folder, "encoders.pkl")

# Load encoders if they exist
encoders = joblib.load(encoders_file) if os.path.exists(encoders_file) else {}

# Load models
models = {}
for mf in model_files:
    safe_name = mf.replace("_model.pkl", "")
    model_path = os.path.join(models_folder, mf)
    models[safe_name] = joblib.load(model_path)
    print(f"✅ Loaded model for: {safe_name}")

# ========== Step 3: Make predictions ==========
predictions = pd.DataFrame(index=df_new.index)

for target, model in models.items():
    print(f"\n🔹 Predicting: {target}")

    X = df_new.copy()

    # Drop target column if it exists
    if target in X.columns:
        X = X.drop(columns=[target])

    # Keep only columns present in training data
    model_features = model.feature_names_in_
    X = X.reindex(columns=model_features, fill_value=0)

    # Predict
    preds = model.predict(X)

    # Decode if categorical
    if target in encoders:
        le = encoders[target]
        preds = le.inverse_transform(preds.astype(int))

    predictions[target] = preds

# ========== Step 4: Output predictions ==========
print("\n🎯 Predictions:")
print(predictions)

# Save predictions
output_file = "predictions.csv"
predictions.to_csv(output_file, index=False)
print(f"\n✅ Predictions saved → {output_file}")
