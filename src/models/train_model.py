import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import yaml
import os

def main():
    params = yaml.safe_load(open("params.yaml"))["model"]

    X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

    best_params = joblib.load("models/best_params.pkl")

    model = RandomForestRegressor(random_state=params["random_state"], **best_params)
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

if __name__ == "__main__":
    main()

