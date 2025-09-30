import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import yaml
import os

def main():
    params = yaml.safe_load(open("params.yaml"))["model"]

    X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

    model = RandomForestRegressor(random_state=params["random_state"])
    grid = GridSearchCV(
        model, params["grid_params"], cv=3, scoring="r2", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(grid.best_params_, "models/best_params.pkl")

if __name__ == "__main__":
    main()

