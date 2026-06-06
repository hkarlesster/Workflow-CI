import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import mlflow
import dagshub
import os
import joblib
import json
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.impute import KNNImputer
import missingno as msno
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from warnings import filterwarnings
warnings.filterwarnings("ignore")

# Create a new MLflow Experiment
mlflow.set_experiment("Heart Disease Prediction Experiment")

data = pd.read_csv("heart_disease_preprocessed.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Heart Disease Status", axis=1),
    data["Heart Disease Status"],
    random_state=42,
    test_size=0.2
)

# Hyperparameter tuning
param_grid = {
    "n_estimators": [100, 300, 505],
    "max_depth": [10, 20, 37]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

# Ambil model terbaik dari Grid Search
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

with mlflow.start_run():
    
    # Log best parameters
    mlflow.log_param("n_estimators", best_params["n_estimators"])
    mlflow.log_param("max_depth", best_params["max_depth"])

    # Train ulang (opsional karena GridSearch sudah fit), kita bisa langsung pakai best_model
    best_model.fit(X_train, y_train)

    # Evaluasi
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    os.makedirs("model", exist_ok=True)
    plt.savefig("model/training_confusion_matrix.png")
    mlflow.log_artifact("model/training_confusion_matrix.png")    

    with open("model/metric_info.json", "w") as f:
        json.dump({"accuracy": acc}, f)
    mlflow.log_artifact("model/metric_info.json")

    joblib.dump(best_model, "model/model.pkl")
    mlflow.log_artifact("model/model.pkl")

    with open("model/requirements.txt", "w") as f:
        f.write("scikit-learn==1.2.2\nmlflow\njoblib\nmatplotlib\npandas\n")
    mlflow.log_artifact("model/requirements.txt")

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model"        
    )  
