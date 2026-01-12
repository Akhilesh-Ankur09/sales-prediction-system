import joblib
import pandas as pd


def load_model():
    return joblib.load("models/rossmann_sales_model.pkl")


def predict_sales(input_data: pd.DataFrame):
    model = load_model()
    prediction = model.predict(input_data)
    return prediction


if __name__ == "__main__":
    # Example input
    sample = pd.read_csv("data/processed/X_test.csv").head(5)
    preds = predict_sales(sample)
    print("Sample predictions:")
    print(preds)
