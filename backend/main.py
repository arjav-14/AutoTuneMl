




from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import os
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.inspection import permutation_importance

from preprocessing import preprocess_data

app = FastAPI()

# =========================
# GLOBAL STATE
# =========================

uploaded_df = None
trained_model = None
problem_type_global = None
leaderboard_global = {}

MODEL_PATH = "models/best_model.pkl"
PREPROCESS_PATH = "models/preprocessing.pkl"

os.makedirs("models", exist_ok=True)


# =========================
# HOME
# =========================

@app.get("/")
def home():
    return {"message": "AutoML backend running 🚀"}


# =========================
# UPLOAD DATASET
# =========================

@app.post("/upload-dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    global uploaded_df
    uploaded_df = pd.read_csv(file.file)

    return {
        "columns": uploaded_df.columns.tolist(),
        "message": "Dataset uploaded successfully"
    }


# =========================
# SHOW DATASET
# =========================

@app.get("/show-dataset/")
def show_dataset():
    return {"rows": uploaded_df.head(50).to_dict(orient="records")}


# =========================
# DATASET SUMMARY
# =========================

@app.get("/dataset-summary/")
def dataset_summary():
    return uploaded_df.describe(include="all").fillna("").to_dict()


# =========================
# CORRELATION MATRIX
# =========================

@app.get("/correlation-matrix/")
def correlation_matrix():
    numeric_df = uploaded_df.select_dtypes(include=["int64", "float64"])
    return {"correlation_matrix": numeric_df.corr().to_dict()}


# =========================
# TRAIN MODEL
# =========================

@app.post("/select-target/")
def select_target(target_column: str):

    global trained_model
    global problem_type_global
    global leaderboard_global

    if target_column not in uploaded_df.columns:
        return {"error": "Invalid target column"}

    processed_df, preprocess_obj = preprocess_data(uploaded_df, target_column)

    X = processed_df.drop(columns=[target_column])
    y = processed_df[target_column]

    preprocess_obj["target_column"] = target_column
    preprocess_obj["feature_columns"] = X.columns.tolist()

    joblib.dump(preprocess_obj, PREPROCESS_PATH)

    if y.dtype == "object":
        problem_type_global = "classification"

        models = {
            "LogisticRegression": LogisticRegression(max_iter=200),
            "RandomForestClassifier": RandomForestClassifier(),
            "SVC": SVC()
        }

        scoring_fn = accuracy_score

    else:
        problem_type_global = "regression"

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(),
            "SVR": SVR()
        }

        scoring_fn = r2_score

    leaderboard_global = {}

    best_score = -999
    best_model = None

    for name, model in models.items():

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        score = scoring_fn(y_test, preds)

        leaderboard_global[name] = score

        if score > best_score:
            best_score = score
            best_model = model

    trained_model = best_model

    joblib.dump(best_model, MODEL_PATH)

    return {
        "problem_type": problem_type_global,
        "metrics": leaderboard_global
    }


# =========================
# LEADERBOARD
# =========================

@app.get("/leaderboard/")
def leaderboard():
    return {"scores": leaderboard_global}


# =========================
# MODEL METRICS
# =========================

@app.get("/model-metrics/")
def model_metrics():

    if trained_model is None:
        return {"error": "Train model first"}

    preprocess = joblib.load(PREPROCESS_PATH)

    target_column = preprocess["target_column"]

    processed_df, _ = preprocess_data(uploaded_df, target_column)

    X = processed_df.drop(columns=[target_column])
    y = processed_df[target_column]

    preds = trained_model.predict(X)

    if problem_type_global == "classification":

        return {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, average="weighted"),
            "recall": recall_score(y, preds, average="weighted"),
            "f1_score": f1_score(y, preds, average="weighted")
        }

    else:

        return {
            "r2_score": r2_score(y, preds),
            "mae": mean_absolute_error(y, preds),
            "rmse": np.sqrt(mean_squared_error(y, preds))
        }


# =========================
# CONFUSION MATRIX
# =========================

@app.get("/confusion-matrix/")
def confusion_matrix_api():

    if problem_type_global != "classification":
        return {"error": "Only available for classification"}

    preprocess = joblib.load(PREPROCESS_PATH)

    target_column = preprocess["target_column"]

    processed_df, _ = preprocess_data(uploaded_df, target_column)

    X = processed_df.drop(columns=[target_column])
    y = processed_df[target_column]

    cm = confusion_matrix(y, trained_model.predict(X))

    return {"confusion_matrix": cm.tolist()}


# =========================
# FEATURE IMPORTANCE
# =========================

@app.get("/feature-importance/")
def feature_importance():

    preprocess = joblib.load(PREPROCESS_PATH)

    target_column = preprocess["target_column"]

    processed_df, _ = preprocess_data(uploaded_df, target_column)

    X = processed_df.drop(columns=[target_column])
    y = processed_df[target_column]

    if hasattr(trained_model, "feature_importances_"):
        importance = trained_model.feature_importances_

    elif hasattr(trained_model, "coef_"):
        importance = trained_model.coef_

    else:
        result = permutation_importance(
            trained_model, X, y, n_repeats=5
        )
        importance = result.importances_mean

    return dict(zip(X.columns, importance.tolist()))


# =========================
# PREDICTION SCHEMA
# =========================

@app.get("/prediction-schema/")
def prediction_schema():

    preprocess = joblib.load(PREPROCESS_PATH)

    feature_columns = preprocess["feature_columns"]

    example = uploaded_df[feature_columns].iloc[0].to_dict()

    return {"prediction_template": example}


# =========================
# PREDICT
# =========================

@app.post("/predict/")
def predict(data: dict):

    preprocess = joblib.load(PREPROCESS_PATH)

    df = pd.DataFrame([data])

    processed_df, _ = preprocess_data(
        pd.concat([uploaded_df, df]),
        preprocess["target_column"]
    )

    X = processed_df.tail(1).drop(columns=[preprocess["target_column"]])

    prediction = trained_model.predict(X)

    return {"prediction": prediction.tolist()}


# =========================
# EXPLAIN (SHAP)
# =========================

@app.post("/explain/")
def explain(data: dict):

    if trained_model is None:
        return {"error": "Train model first"}

    preprocess = joblib.load(PREPROCESS_PATH)

    target_column = preprocess["target_column"]

    df_input = pd.DataFrame([data])

    temp_df = pd.concat([uploaded_df, df_input], ignore_index=True)

    processed_df, _ = preprocess_data(temp_df, target_column)

    X = processed_df.tail(1).drop(columns=[target_column])

    try:
        explainer = shap.Explainer(trained_model, processed_df.drop(columns=[target_column]))
        shap_values = explainer(X)

        explanation = dict(
            zip(X.columns, shap_values.values[0].tolist())
        )

    except:
        explanation = "SHAP not supported for this model"

    prediction = trained_model.predict(X)

    return {
        "prediction": prediction.tolist(),
        "explanation": explanation
    }


# =========================
# DATASET HEALTH
# =========================

@app.get("/dataset-health/")
def dataset_health():

    if uploaded_df is None:
        return {"error": "Upload dataset first"}

    return {
        "rows": uploaded_df.shape[0],
        "columns": uploaded_df.shape[1],
        "missing_values": uploaded_df.isnull().sum().to_dict(),
        "duplicate_rows": int(uploaded_df.duplicated().sum()),
        "memory_usage_MB": round(
            uploaded_df.memory_usage(deep=True).sum() / 1024**2, 2
        )
    }


# =========================
# MODEL VERSION HISTORY
# =========================

@app.get("/model-version-history/")
def model_versions():

    return {
        "latest_model": MODEL_PATH,
        "problem_type": problem_type_global
    }


# =========================
# DOWNLOAD MODEL
# =========================

@app.get("/download-model/")
def download_model():
    return FileResponse(MODEL_PATH)


# =========================
# DOWNLOAD PIPELINE
# =========================

@app.get("/download-pipeline/")
def download_pipeline():
    return FileResponse(PREPROCESS_PATH)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)