import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

def main():
    X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed_data/y_test.csv")

    model = joblib.load("models/model.pkl")
    preds = model.predict(X_test)

    # Save predictions
    os.makedirs("data/processed_data", exist_ok=True)
    pd.DataFrame({"y_true": y_test.values.ravel(), "y_pred": preds}).to_csv(
        "data/processed_data/predictions.csv", index=False
    )

    # Save metrics
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump({"mse": mse, "r2": r2}, f, indent=4)

if __name__ == "__main__":
    main()

