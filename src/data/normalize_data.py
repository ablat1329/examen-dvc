import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import yaml
import os

def main():
    params = yaml.safe_load(open("params.yaml"))["scaling"]

    input_dir = "data/processed_data"
    output_dir = "data/processed_data"
    os.makedirs(output_dir, exist_ok=True)

    X_train = pd.read_csv(f"{input_dir}/X_train.csv")
    X_test = pd.read_csv(f"{input_dir}/X_test.csv")

    # Choose scaler based on params
    if params["method"] == "standard":
        scaler = StandardScaler()
    elif params["method"] == "minmax":
        scaler = MinMaxScaler()
    elif params["method"] == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {params['method']}")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
        f"{output_dir}/X_train_scaled.csv", index=False
    )
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(
        f"{output_dir}/X_test_scaled.csv", index=False
    )

if __name__ == "__main__":
    main()

