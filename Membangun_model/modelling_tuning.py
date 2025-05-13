import pandas as pd
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
import mlflow
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import json

mlflow.set_tracking_uri("https://dagshub.com/ikrar557/mlsystem-dicoding.mlflow")
mlflow.set_experiment("titanic_tuning_experiment")

df = pd.read_csv("../preprocessing/titanic_preprocessing.csv")

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
    
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

with mlflow.start_run():

    mlflow.sklearn.log_model(best_model, "model")
    
    mlflow.log_metric("accuracy", accuracy)
    
    mlflow.log_metric("test_accuracy_score", test_accuracy)
    mlflow.log_metric("test_precision_score", test_precision)
    mlflow.log_metric("test_f1_score", test_f1)
    mlflow.log_metric("test_recall_score", test_recall)

    mlflow.log_metric("training_accuracy_score", train_accuracy)
    mlflow.log_metric("training_precision_score", train_precision)
    mlflow.log_metric("training_f1_score", train_f1)
    mlflow.log_metric("training_recall_score", train_recall)
    
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", grid_search.best_params_["n_estimators"])
    mlflow.log_param("max_depth", grid_search.best_params_["max_depth"])
    mlflow.log_param("best_params", grid_search.best_params_)
    mlflow.log_param("cv_results", grid_search.cv_results_)

    cm_list = cm.tolist()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
    plt.title("Confusion Matrix")
    plt.savefig("training_confusion_matrix.png")
    plt.close()
    
    mlflow.log_artifact("training_confusion_matrix.png")

    cv_results_serialized = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in grid_search.cv_results_.items()
    }
    
    estimator_info = {
        "best_params": grid_search.best_params_,
        "cv_results": cv_results_serialized
    }
    
    with open("estimator.html", "w") as f:
        f.write(json.dumps(estimator_info, indent=4))

    mlflow.log_artifact("estimator.html")
    
    with open("confusion_matrix.json", "w") as f:
        json.dump(cm_list, f)

    mlflow.log_artifact("confusion_matrix.json")