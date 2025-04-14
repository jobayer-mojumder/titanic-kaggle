import pandas as pd
import warnings
import scipy.sparse as sp
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

PARAM_GRIDS = {
    "dt": {
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 2],
    },
    "rf": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 2],
    },
    "xgb": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
    "lgbm": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, -1],
        "num_leaves": [15, 31],
        "learning_rate": [0.01, 0.1],
    },
    "cb": {
        "iterations": [100, 200],
        "depth": [3, 5],
        "learning_rate": [0.01, 0.1],
        "l2_leaf_reg": [1, 3],
    },
}

# Define base models
BASE_MODELS = {
    "dt": DecisionTreeClassifier(random_state=42),
    "rf": RandomForestClassifier(random_state=42),
    "xgb": XGBClassifier(eval_metric="logloss", random_state=42),
    "lgbm": LGBMClassifier(random_state=42, verbose=-1),
    "cb": CatBoostClassifier(verbose=0, random_seed=42),
}


def ensure_numeric_features(X):
    """Ensure all columns are numeric; expand sparse matrices if found."""
    X = pd.DataFrame(X).copy()

    for col in X.columns:
        if isinstance(X[col].iloc[0], sp.spmatrix):
            print(f"🔄 Expanding sparse matrix column: {col}")
            expanded = pd.DataFrame(
                np.vstack(X[col].apply(lambda x: x.toarray().ravel())), index=X.index
            )
            expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]
            X = pd.concat([X.drop(columns=[col]), expanded], axis=1)

    non_numeric = X.select_dtypes(include=["object", "category"]).columns
    if len(non_numeric) > 0:
        print(f"🔄 Auto-encoding non-numeric columns: {list(non_numeric)}")
        X = pd.get_dummies(X, columns=non_numeric, drop_first=True)

    return X


def tune_model(X, y, model_key, cv=5, scoring="accuracy"):
    print(f"🔍 Tuning model: {model_key.upper()}")

    model = BASE_MODELS[model_key]
    param_grid = PARAM_GRIDS[model_key]

    X = ensure_numeric_features(X)

    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, verbose=1, n_jobs=-1)

    grid.fit(X, y)

    print(f"✅ Best params: {grid.best_params_}")
    print(f"📈 Best score: {grid.best_score_:.5f}")

    return grid.best_estimator_, grid.best_params_
