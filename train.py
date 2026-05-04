import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error

# ---------------- Load Data ----------------
df = pd.read_csv("data/BostonHousing.csv")

# ---------------- Handle Missing Values ----------------
imputer = SimpleImputer(strategy="mean")
df[["rm"]] = imputer.fit_transform(df[["rm"]])

# ---------------- Features & Target ----------------
FEATURES = ["rm", "lstat", "ptratio", "nox", "dis", "tax", "age"]

X = df[FEATURES]
y = df["medv"]

# ---------------- Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Model ----------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- Evaluation ----------------
y_pred = model.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# ---------------- Save Model + Features ----------------
os.makedirs("models", exist_ok=True)

joblib.dump({
    "model": model,
    "features": FEATURES
}, "models/model.pkl")

print("Model saved successfully")