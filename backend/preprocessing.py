import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def preprocess_data(df, target_column):

    df = df.copy()

    if target_column not in df.columns:
        raise ValueError("Target column not found in dataset")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    preprocessing_objects = {
        "label_encoders": {},
        "target_encoder": None,
        "num_imputer": None,
        "cat_imputer": None,
        "scaler": None,
        "feature_columns": X.columns.tolist(),
        "target_column": target_column
    }

    # Detect column types
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns


    # -----------------------------
    # Handle missing numeric values
    # -----------------------------
    if len(numerical_cols) > 0:

        num_imputer = SimpleImputer(strategy="median")

        X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])

        preprocessing_objects["num_imputer"] = num_imputer


    # -----------------------------
    # Handle missing categorical values
    # -----------------------------
    if len(categorical_cols) > 0:

        cat_imputer = SimpleImputer(strategy="most_frequent")

        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

        preprocessing_objects["cat_imputer"] = cat_imputer


    # -----------------------------
    # Encode categorical feature columns
    # -----------------------------
    label_encoders = {}

    for col in categorical_cols:

        le = LabelEncoder()

        X[col] = le.fit_transform(X[col])

        label_encoders[col] = le

    preprocessing_objects["label_encoders"] = label_encoders


    # -----------------------------
    # Scale numeric columns
    # -----------------------------
    if len(numerical_cols) > 0:

        scaler = StandardScaler()

        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

        preprocessing_objects["scaler"] = scaler


    # -----------------------------
    # Encode target column (if classification)
    # -----------------------------
    if y.dtype == "object":

        target_encoder = LabelEncoder()

        y = target_encoder.fit_transform(y)

        preprocessing_objects["target_encoder"] = target_encoder


    # Combine processed X and y
    processed_df = pd.concat([X, pd.Series(y, name=target_column)], axis=1)

    return processed_df, preprocessing_objects    

