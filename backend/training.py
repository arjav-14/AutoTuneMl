import optuna
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC


MODEL_PATH = "models/best_model.pkl"


def detect_problem_type(y):

    if y.dtype == "object" or len(np.unique(y)) < 20:
        return "classification"

    return "regression"


# --------------------------------------------------
# OPTUNA REGRESSION TUNING
# --------------------------------------------------

def tune_rf_regressor(X_train, X_test, y_train, y_test):

    def objective(trial):

        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        return r2_score(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_model = RandomForestRegressor(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)

    return best_model


# --------------------------------------------------
# OPTUNA CLASSIFICATION TUNING
# --------------------------------------------------

def tune_rf_classifier(X_train, X_test, y_train, y_test):

    def objective(trial):

        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        return accuracy_score(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_model = RandomForestClassifier(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)

    return best_model


# --------------------------------------------------
# MAIN TRAINING FUNCTION
# --------------------------------------------------

def train_models(df):

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    problem_type = detect_problem_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {}

    if problem_type == "regression":

        models["LinearRegression"] = LinearRegression()
        models["SVR"] = SVR()
        models["RandomForest_Optimized"] = tune_rf_regressor(
            X_train, X_test, y_train, y_test
        )

        scoring_function = r2_score

    else:

        models["LogisticRegression"] = LogisticRegression(max_iter=1000)
        models["SVC"] = SVC()
        models["RandomForest_Optimized"] = tune_rf_classifier(
            X_train, X_test, y_train, y_test
        )

        scoring_function = accuracy_score


    results = {}
    best_score = -999
    best_model = None

    for name, model in models.items():

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        score = scoring_function(y_test, preds)

        results[name] = score

        if score > best_score:

            best_score = score
            best_model = model


    joblib.dump(best_model, MODEL_PATH)

    return best_model, problem_type, results, X.columns.tolist()