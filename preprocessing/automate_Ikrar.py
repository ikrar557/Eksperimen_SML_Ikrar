import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def preprocess_data(file_path):

    df = pd.read_csv(file_path)

    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_condition = (df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR))
    df = df[~outlier_condition]

    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    bins = [0, 12, 18, 30, 50, 100]
    labels = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    df['Age_Group'] = LabelEncoder().fit_transform(df['Age_Group'].astype(str))

    output_path = "preprocessing/titanic_preprocessing.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    ## test trigger 3
    print(f"Data preprocessed and saved to: {output_path}")
    return df

if __name__ == "__main__":
    preprocess_data("titanic_raw.csv")
