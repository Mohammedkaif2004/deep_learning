import os
import sys
from dataclasses import dataclass
import joblib

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.utils import evaluate_models


@dataclass
class MlModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'ml_model.pkl')


class MlModelTrainer:
    def __init__(self):
        self.config = MlModelTrainerConfig()
        self.data_transformation = DataTransformation()

    def initiate_model_trainer(self):
        try:
            logging.info("Starting data transformation for model training")
            X_train_transformed, y_train, X_test_transformed, y_test, preprocessor = (
                self.data_transformation.initiate_data_transformation()
            )

            logging.info("Creating model list")
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "XGBoost": XGBClassifier(),
                "Support Vector Machine": SVC(),
                "Naive Bayes": GaussianNB(),
            }

            params = {
                "Logistic Regression": {},
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "XGBoost": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "AdaBoost": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "K-Nearest Neighbors": {},
                "Support Vector Machine": {},
                "Naive Bayes": {}
            }

            logging.info("Evaluating models")
            model_report = evaluate_models(
                X_train=X_train_transformed,
                y_train=y_train,
                X_test=X_test_transformed,
                y_test=y_test,
                models=models,
                param=params
            )

            # Choose the metric to select the best model (here we use Accuracy)
            best_model_name = max(model_report, key=lambda k: model_report[k]["Accuracy"])
            best_model_score = model_report[best_model_name]["Accuracy"]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name}")
            logging.info(f"Accuracy: {best_model_score:.4f}")
            logging.info(f"R² Score: {model_report[best_model_name]['R2 Score']:.4f}")

            # Save the best model
            joblib.dump(best_model, self.config.trained_model_file_path)
            logging.info(f"Saved best model to {self.config.trained_model_file_path}")

            # Evaluate and log final test performance
            predicted = best_model.predict(X_test_transformed)
            r2 = r2_score(y_test, predicted)
            logging.info(f"Final R2 score on test set: {r2:.4f}")

            return {
                "best_model": best_model_name,
                "accuracy": best_model_score,
                "r2_score": r2
            }

        except Exception as e:
            logging.error(f"Error occurred during model training: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = MlModelTrainer()
    result = trainer.initiate_model_trainer()
    logging.info(f"Training complete. Best model: {result['best_model']} | Accuracy: {result['accuracy']:.4f} | R²: {result['r2_score']:.4f}")

         