import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ========== Step 1: Load both Excel files ==========
file1 = "ceirrus.xlsx"
file2 = "subtallis.xlsx"

df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# ========== Step 2: Align columns ==========
if not df1.columns.equals(df2.columns):
    print("⚠️ Columns differ. Aligning columns before merging...")
    all_cols = sorted(set(df1.columns).union(set(df2.columns)))
    df1 = df1.reindex(columns=all_cols)
    df2 = df2.reindex(columns=all_cols)

# Merge datasets
data = pd.concat([df1, df2], ignore_index=True)
print(f"📊 Final dataset shape: {data.shape}")

# ========== Step 3: Clean dataset ==========
# Drop useless columns
drop_cols = [c for c in data.columns if "Unnamed" in str(c) or data[c].isna().all() or data[c].nunique() <= 1]
if drop_cols:
    print(f"🗑️ Dropping useless columns: {drop_cols}")
    data = data.drop(columns=drop_cols)

# Create models folder
os.makedirs("models", exist_ok=True)

# Store encoders for categorical targets
encoders = {}

# ========== Step 4: Train model for each target ==========
for target in data.columns:
    print(f"\n🔹 Training model for target: {target}")

    X = data.drop(columns=[target])
    y = data[target]

    # Handle missing values in features
    X = X.fillna(0)

    # Handle target values
    if y.isnull().any():
        if y.dtype == "object":
            y = y.fillna("missing")
        else:
            y = y.fillna(y.mean())

    # Mixed type fix: force uniform type
    if y.dtype == "object" or y.apply(lambda v: isinstance(v, str)).any():
        y = y.astype(str)  # force to string
        le = LabelEncoder()
        y = le.fit_transform(y)
        encoders[target] = le
        model = RandomForestClassifier(random_state=42)
    else:
        y = pd.to_numeric(y, errors="coerce").fillna(0)
        model = RandomForestRegressor(random_state=42)

    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    except ValueError as e:
        print(f"⚠️ Skipping {target} (error: {e})")
        continue

    # Train model
    model.fit(X_train, y_train)

    # Save model
    safe_target = str(target).replace("/", "_").replace("\\", "_").replace(" ", "_")
    model_filename = f"models/{safe_target}_model.pkl"
    joblib.dump(model, model_filename)
    print(f"✅ Saved {target} model → {model_filename}")

# Save encoders
joblib.dump(encoders, "models/encoders.pkl")
print("\n🎉 Training complete! All models + encoders saved in 'models/' folder.")
