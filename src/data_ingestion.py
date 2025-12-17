import pandas as pd
import os
import logging

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

# Logging configuration
logging.basicConfig(
    filename="logs/data_ingestion.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def ingest_data():
    try:

        df = pd.read_csv("../data/raw/churn.csv")

        logging.info("Raw data loaded successfully")
        logging.info(f"Data shape: {df.shape}")

        print("Data loaded successfully")
        print("Shape:", df.shape)

        return df

    except Exception as e:
        logging.error("Error during data ingestion")
        logging.error(str(e))
        raise e


if __name__ == "__main__":
    ingest_data()
