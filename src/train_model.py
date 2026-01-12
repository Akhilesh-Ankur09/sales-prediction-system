import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_model():
    print("Loading processed training data...")

    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    print("Training RandomForest model (memory-optimized)...")

    model = RandomForestRegressor(
        n_estimators=80,          # reduced from 200
        max_depth=20,            # limit tree depth
        min_samples_leaf=20,     # reduce overfitting & memory
        n_jobs=2,                # limit CPU usage
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Model MAE: {mae:.2f}")
    print(f"Model R2 Score: {r2:.4f}")

    print("Saving trained model...")
    joblib.dump(model, "models/rossmann_sales_model.pkl", compress=3)

    print("Model saved successfully to: models/rossmann_sales_model.pkl")


if __name__ == "__main__":
    train_model()
