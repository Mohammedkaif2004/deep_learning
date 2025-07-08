import sys
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            try:
                logging_msg = f"Evaluating model: {model_name}"
                print(logging_msg)  # Optional debug print
                params = param.get(model_name, {})

                if params:
                    gs = GridSearchCV(model, params, cv=3, n_jobs=-1)
                    gs.fit(X_train, y_train)
                    model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Convert probabilities to class labels if needed
                if y_test_pred.dtype != int and y_test_pred.ndim == 1:
                    y_test_pred = (y_test_pred >= 0.5).astype(int)
                    y_train_pred = (y_train_pred >= 0.5).astype(int)

                train_model_r2 = r2_score(y_train, y_train_pred)
                test_model_r2 = r2_score(y_test, y_test_pred)
                test_model_accuracy = accuracy_score(y_test, y_test_pred)

                report[model_name] = {
                    "R2 Score": test_model_r2,
                    "Accuracy": test_model_accuracy
                }

            except Exception as inner_e:
                print(f"Skipping model {model_name} due to error: {inner_e}")
                continue

        return report

    except Exception as e:
        raise CustomException(e, sys)
