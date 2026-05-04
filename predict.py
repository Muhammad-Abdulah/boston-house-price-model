import pandas as pd
import joblib

# correct path
data = joblib.load("../models/model.pkl")

model = data["model"]
features = data["features"]

sample = pd.DataFrame([[
    5.7, 11.2, 21, 0.5, 3.7, 300, 70
]], columns=features)

prediction = model.predict(sample)

print("Predicted House Price:", prediction[0])