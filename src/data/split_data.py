import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

def main():
    # Load params
    params = yaml.safe_load(open("params.yaml"))["split"]
    # params = {}
    # params["test_size"]=0.2
    # params["random_state"]=42
    input_path = "data/raw_data/raw.csv"
    output_dir = "data/processed_data"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path,index_col=0)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"]
    )

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

if __name__ == "__main__":
    main()
    # input_path = "../../data/raw_data/raw.csv"
    # output_dir = "../../data/processed_data"
    # os.makedirs(output_dir, exist_ok=True)

    # df = pd.read_csv(input_path,index_col=0)
    # X = df.iloc[:, :-1]
    # y = df.iloc[:, -1]


