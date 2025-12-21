import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


def preprocess_data(
    input_path: str,
    output_dir: str
):
    df = pd.read_csv(input_path)

    df = df.drop(columns=['customerID'])

    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce'
    )
    df = df.dropna()

    df = df.drop_duplicates()

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    numerical_features = X.select_dtypes(
        include=['int64', 'float64']
    ).columns

    categorical_features = X.select_dtypes(
        include=['object']
    ).columns

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(X_train_processed, os.path.join(output_dir, 'X_train.pkl'))
    joblib.dump(X_test_processed, os.path.join(output_dir, 'X_test.pkl'))
    joblib.dump(y_train, os.path.join(output_dir, 'y_train.pkl'))
    joblib.dump(y_test, os.path.join(output_dir, 'y_test.pkl'))
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))

    print("Preprocessing selesai. Output berhasil disimpan.")


if __name__ == "__main__":
    INPUT_PATH = "Telco-Customer-Churn_raw/Telco-Customer-Churn.csv"
    OUTPUT_DIR = "preprocessing/Telco-Customer-Churn_clean"

    preprocess_data(
        input_path=INPUT_PATH,
        output_dir=OUTPUT_DIR
    )
