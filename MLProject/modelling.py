# modelling/modelling.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

mlflow.sklearn.autolog()

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("titanic_experiment")

df = pd.read_csv("../preprocessing/titanic_preprocessing.csv")

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)

with mlflow.start_run():
    model.fit(X_train, y_train)
    
    mlflow.sklearn.log_model(model, "model")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy}")

