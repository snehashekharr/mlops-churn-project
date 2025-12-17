import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split

# Ensure processed and logs directories exist
if not os.path.exists("../data/processed"):
    os.makedirs("../data/processed")
if not os.path.exists("../logs"):
    os.makedirs("../logs")

# Logging setup
logging.basicConfig(
    filename="../logs/data_preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def preprocess_data():
    try:
        # Load raw data
        df = pd.read_csv("../data/raw/churn.csv")
        logging.info("Raw data loaded successfully for preprocessing")

        # Drop rows with missing values
        df = df.dropna()
        logging.info(f"Data shape after dropping nulls: {df.shape}")

        # Encode categorical columns (works even if data is small)
        for col in ["gender", "churn"]:
            df[col] = df[col].map({'Male': 0, 'Female': 1, 'No': 0, 'Yes': 1})
        logging.info("Categorical columns encoded")

        # Split into train/test
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        # Save processed data
        train_df.to_csv("../data/processed/train.csv", index=False)
        test_df.to_csv("../data/processed/test.csv", index=False)
        logging.info("Processed data saved successfully")

        print("Data preprocessing completed successfully!")

    except Exception as e:
        logging.error("Error during data preprocessing")
        logging.error(str(e))
        raise e

if __name__ == "__main__":
    preprocess_data()
