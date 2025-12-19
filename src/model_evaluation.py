import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model():
    test_path = "../data/processed/test.csv"
    model_path = "../models/churn_model.pkl"

    # load test data
    df = pd.read_csv(test_path)
    X_test = df.drop("churn", axis=1)
    y_test = df["churn"]

    # load trained model (ONLY joblib)
    model = joblib.load(model_path)
    print("Model type:", type(model))  # sanity check

    # predictions
    y_pred = model.predict(X_test)

    # metrics
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()
