import os
import sys
from dataclasses import dataclass

import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping

from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation


@dataclass
class DeepLearningModelTrainerConfig:
    model_save_path: str = os.path.join("artifacts", "best_dl_model.h5")
    tuner_dir: str = os.path.join("artifacts", "keras_tuner")


class DeepLearningModelTrainer:
    def __init__(self):
        self.config = DeepLearningModelTrainerConfig()
        self.data_transformer = DataTransformation()

    def build_model(self, hp):
        try:
            model = keras.Sequential()
            model.add(layers.Input(shape=(self.input_shape,)))

            for i in range(hp.Int('num_layers', 2, 5)):
                model.add(
                    layers.Dense(
                        units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                        activation='relu'
                    )
                )

            model.add(layers.Dense(1, activation='sigmoid'))  # For binary classification

            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
                ),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            return model

        except Exception as e:
            logging.error("Error occurred while building the Keras model")
            raise CustomException(e, sys)

    def initiate_deep_learning_training(self):
        try:
            logging.info("Starting deep learning model training...")

            # Load transformed data
            X_train, y_train, X_test, y_test, _ = self.data_transformer.initiate_data_transformation()
            self.input_shape = X_train.shape[1]

            logging.info(f"Data loaded. Input shape: {self.input_shape}")

            tuner = RandomSearch(
                self.build_model,
                objective='val_accuracy',
                max_trials=5,
                executions_per_trial=2,
                directory=self.config.tuner_dir,
                project_name="churn_dl_tuning"
            )

            logging.info("Starting hyperparameter tuning with RandomSearch...")

            tuner.search(
                X_train, y_train,
                epochs=50,
                validation_data=(X_test, y_test),
                callbacks=[EarlyStopping(monitor='val_loss', patience=3)]
            )

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            logging.info(f"Best hyperparameters found: {best_hps.values}")

            # Build and train final model
            best_model = tuner.hypermodel.build(best_hps)
            history = best_model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
            )

            # Evaluate and save model
            loss, accuracy = best_model.evaluate(X_test, y_test)
            logging.info(f"Final model evaluation - Loss: {loss}, Accuracy: {accuracy}")

            best_model.save(self.config.model_save_path)
            logging.info(f"Model saved at {self.config.model_save_path}")

            return accuracy

        except Exception as e:
            logging.error("Error during deep learning model training")
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = DeepLearningModelTrainer()
    accuracy = trainer.initiate_deep_learning_training()
    logging.info(f"Deep learning training complete. Final test accuracy: {accuracy}")
