import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import os
import warnings

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

BASE_MODELS = {
    "dt": DecisionTreeClassifier(random_state=42),
    "rf": RandomForestClassifier(random_state=42),
    "xgb": XGBClassifier(eval_metric="logloss", random_state=42),
    "lgbm": LGBMClassifier(random_state=42, verbose=-1),
    "cb": CatBoostClassifier(verbose=0, random_seed=42),
}


def tune_model(X, y, model_key, cv=5, scoring="accuracy"):
    print(f"🔍 Tuning model: {model_key.upper()}")

    model = BASE_MODELS[model_key]
    param_grid = PARAM_GRIDS[model_key]

    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, verbose=1, n_jobs=-1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        grid.fit(X, y)

    print(f"✅ Best params: {grid.best_params_}")
    print(f"📈 Best score: {grid.best_score_:.5f}")

    # Save tuning results to CSV results/ folder, if not created, create it
    results_dir = "results/tuning"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{model_key}_tuning_results.csv")

    if results_file:
        df = pd.DataFrame(grid.cv_results_)
        df.to_csv(results_file, index=False)
        print(f"🗘️ Saved tuning results to {results_file}")
    else:
        print("⚠️ No tuning results to save.")

    return grid.best_estimator_
