import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (simple example)
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Feature
y = 2.5 * X + np.random.randn(100, 1) * 2  # Target with noise

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter options
n_estimators_list = [10, 50, 100]
max_depth_list = [None, 5, 10]

best_mse = float("inf")  # Track best model

for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        with mlflow.start_run():
            # Train model with different hyperparameters
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train.ravel())

            # Predictions
            y_pred = model.predict(X_test)

            # Evaluate model
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log parameters & metrics
            mlflow.log_param("model", "RandomForestRegressor")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2_score", r2)

            # Log model
            mlflow.sklearn.log_model(model, "random_forest_model")

            print(f"Trained model: n_estimators={n_estimators}, max_depth={max_depth}, MSE={mse:.4f}, R2={r2:.4f}")

            # Save best model
            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}

# Log the best model separately
with mlflow.start_run():
    mlflow.log_param("best_n_estimators", best_params["n_estimators"])
    mlflow.log_param("best_max_depth", best_params["max_depth"])
    mlflow.log_metric("best_mse", best_mse)
    mlflow.sklearn.log_model(best_model, "best_random_forest_model")

    print(f"Best Model: n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}, MSE={best_mse:.4f}")

print("Hyperparameter tuning completed. Start MLflow UI to view results.")

