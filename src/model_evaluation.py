import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os


# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rossmann_sales_model.pkl")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports")


def main():
    print("Loading test data and model...")

    X_test = pd.read_csv(os.path.join(DATA_PATH, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_PATH, "y_test.csv")).values.ravel()
    model = joblib.load(MODEL_PATH)

    print("Generating predictions...")
    preds = model.predict(X_test)

    # Ensure reports folder exists
    os.makedirs(REPORT_PATH, exist_ok=True)

    # Actual vs Predicted
    plt.figure()
    plt.scatter(y_test[:1000], preds[:1000])
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.savefig(os.path.join(REPORT_PATH, "actual_vs_predicted.png"))
    plt.close()

    # Error distribution
    errors = y_test - preds
    plt.figure()
    plt.hist(errors, bins=50)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Prediction Error Distribution")
    plt.savefig(os.path.join(REPORT_PATH, "error_distribution.png"))
    plt.close()

    # Feature importance
    importances = model.feature_importances_
    features = X_test.columns

    fi = pd.DataFrame({"Feature": features, "Importance": importances})
    fi = fi.sort_values(by="Importance", ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(fi["Feature"], fi["Importance"])
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_PATH, "feature_importance.png"))
    plt.close()

    print("Evaluation charts saved to reports/ folder.")


if __name__ == "__main__":
    main()
