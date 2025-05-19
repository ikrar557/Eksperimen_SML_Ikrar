import requests
import json
import pandas as pd

df = pd.read_csv("titanic_preprocessing.csv")

if 'Survived' in df.columns:
    df = df.drop(columns=["Survived"])

sample = df.sample(n=1, random_state=None)

data_split = sample.to_dict(orient="split")

response = requests.post(
    url="http://localhost:8000/predict",
    headers={"Content-Type": "application/json"},
    data=json.dumps({"dataframe_split": data_split})
)

print("Data digunakan untuk inferensi:")
print(sample)

print("\nStatus Code:", response.status_code)
try:
    print("Response:", response.json())
except Exception:
    print("Raw Response:", response.text)
