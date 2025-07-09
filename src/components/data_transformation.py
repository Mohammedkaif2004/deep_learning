import numpy as np
import pandas as pd
import os
import sys
import joblib

from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.components.data_ingetion import DataIngestion, DataIngestionConfig
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.data_ingestion = DataIngestion()
        logging.info("DataTransformation instance created")

    def get_data_transformer_object(self, df_sample):
        logging.info("Creating data transformer object")
        try:
            # Determine numerical and categorical columns from sample data
            num_features = df_sample.select_dtypes(exclude="object").columns.tolist()
            cat_features = df_sample.select_dtypes(include="object").columns.tolist()

            logging.info(f"Numerical features: {num_features}")
            logging.info(f"Categorical features: {cat_features}")

            # Create pipelines
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine pipelines in column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, num_features),
                    ('cat', categorical_transformer, cat_features)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.error("Error in get_data_transformer_object")
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        logging.info("Starting data transformation")
        try:
            # Load train and test data
            train_df = pd.read_csv(self.data_ingestion_config.train_data_path)
            test_df = pd.read_csv(self.data_ingestion_config.test_data_path)

            # Drop unnecessary columns
            drop_cols = ['RowNumber', 'CustomerId', 'Surname']
            train_df.drop(columns=drop_cols, inplace=True)
            test_df.drop(columns=drop_cols, inplace=True)

            # Separate features and target
            X_train = train_df.drop(columns=['Exited'])
            y_train = train_df['Exited']
            X_test = test_df.drop(columns=['Exited'])
            y_test = test_df['Exited']

            # Create preprocessor using feature-only data
            preprocessor = self.get_data_transformer_object(X_train)

            # Fit and transform
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Data transformation complete")
            logging.info(f"Transformed training shape: {X_train_transformed.shape}")
            logging.info(f"Transformed test shape: {X_test_transformed.shape}")
            

            # Save preprocessor object
            joblib.dump(preprocessor, self.data_transformation_config.preprocessor_path)
            logging.info(f"Saved preprocessor to {self.data_transformation_config.preprocessor_path}")

            return X_train_transformed, y_train, X_test_transformed, y_test, self.data_transformation_config.preprocessor_path

        except Exception as e:
            logging.error("Error during data transformation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()
    X_train_transformed, y_train, X_test_transformed, y_test , preprocessor_path = obj.initiate_data_transformation()
    logging.info("Data transformation pipeline executed successfully")
    

    
