import os
import pandas as pd
import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

# logging setup
logging.basicConfig(
    filename="logs/model_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_model():
    try:
        logging.info("Starting model training")

        # load training data
        train_path = "../data/processed/train.csv"

        df = pd.read_csv(train_path)

        # separate features and target
        X = df.drop("Churn", axis=1)
        y = df["Churn"]

        # train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        # save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "../models/churn_model.pkl")


        logging.info("Model training completed and saved")

    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise e

if __name__ == "__main__":
    train_model()
