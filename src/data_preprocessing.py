import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    sales = pd.read_csv("data/raw/train.csv", parse_dates=["Date"])
    stores = pd.read_csv("data/raw/store.csv")

    df = sales.merge(stores, on="Store", how="left")
    return df


def preprocess_data(df):
    # Drop columns not useful for prediction
    drop_cols = ["Date", "Sales", "PromoInterval"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Fill missing values
    df = df.fillna(0)

    # Encode categorical features
    categorical_cols = ["StoreType", "Assortment", "StateHoliday"]
    df = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns])

    # Target variable
    target = pd.read_csv("data/raw/train.csv", parse_dates=["Date"])["Sales"]

    return train_test_split(df, target, test_size=0.2, random_state=42)


if __name__ == "__main__":
    print("Loading dataset...")
    df = load_data()

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("Preprocessing completed successfully.")
