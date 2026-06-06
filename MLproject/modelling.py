import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Create a new MLflow Experiment
mlflow.set_experiment("Heart Disease Prediction Experiment")

data = pd.read_csv("heart_disease_preprocessed.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Heart Disease Status", axis=1),
    data["Heart Disease Status"],
    random_state=42,
    test_size=0.2
)

with mlflow.start_run():
    # Log parameters
    n_estimators = 505
    max_depth = 37
    mlflow.autolog()
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"        
    )
    model.fit(X_train, y_train)
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)