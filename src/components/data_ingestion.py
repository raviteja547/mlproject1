import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the DataIngestion method")

        try:
            logging.info("Reading dataset from notebook/data/StudentsPerformance.csv")
            df = pd.read_csv(r"notebook\data\StudentsPerformance.csv")
            logging.info(f"Dataset loaded successfully with shape: {df.shape}")

            logging.info("Performing train-test split (80-20)")
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
            logging.info(f"Train-test split completed - Train: {train_data.shape}, Test: {test_data.shape}")

            logging.info("Creating artifacts directory and saving data files")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Save all files
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info(f"Raw data saved to: {self.ingestion_config.raw_data_path}")
            logging.info(f"Train data saved to: {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved to: {self.ingestion_config.test_data_path}")
            
            logging.info("Data ingestion completed successfully!")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    logging.info("Starting data ingestion script...")
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    logging.info(f"Data ingestion completed - Train path: {train_path}, Test path: {test_path}")