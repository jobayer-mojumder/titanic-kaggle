from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

MAX_FEATURES_NUM = 10

DECISION_TREE_BASE_SCORE = 0.75754
XGBOOST_BASE_SCORE = 0.78547
RANDOM_FOREST_BASE_SCORE = 0.79106
LIGHTGBM_BASE_SCORE = 0.78994
CATBOOST_BASE_SCORE = 0.81006

DECISION_TREE_KAGGLE_SCORE = 0.73205
XGBOOST_KAGGLE_SCORE = 0.75119
RANDOM_FOREST_KAGGLE_SCORE = 0.76555
LIGHTGBM_KAGGLE_SCORE = 0.76555
CATBOOST_KAGGLE_SCORE = 0.77751

BASELINE_SCORE = {
    "dt": DECISION_TREE_BASE_SCORE,
    "xgb": XGBOOST_BASE_SCORE,
    "rf": RANDOM_FOREST_BASE_SCORE,
    "lgbm": LIGHTGBM_BASE_SCORE,
    "cb": CATBOOST_BASE_SCORE,
}

MODEL_ORDER = {
    "Decision Tree": 1,
    "XGBoost": 2,
    "Random Forest": 3,
    "LightGBM": 4,
    "CatBoost": 5,
}

KAGGLE_BASELINE_SCORE = {
    "dt": DECISION_TREE_KAGGLE_SCORE,
    "xgb": XGBOOST_KAGGLE_SCORE,
    "rf": RANDOM_FOREST_KAGGLE_SCORE,
    "lgbm": LIGHTGBM_KAGGLE_SCORE,
    "cb": CATBOOST_KAGGLE_SCORE,
}


DEFAULT_MODELS = {
    "dt": DecisionTreeClassifier(random_state=42),
    "rf": RandomForestClassifier(random_state=42),
    "xgb": XGBClassifier(
        random_state=42,
        eval_metric="logloss",
    ),
    "lgbm": LGBMClassifier(random_state=42, verbose=-1),
    "cb": CatBoostClassifier(random_seed=42, verbose=0),
}

PARAM_GRIDS = {
    "dt": {
        "max_depth": [None, 3, 4, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 5, 10],
    },
    "rf": {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [None, 3, 4, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 5],
    },
    "xgb": {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [5, 6, 7, 10],
        "learning_rate": [0.2, 0.3, 0.4],
        "reg_lambda": [1, 5, 10],
        "reg_alpha": [0, 2, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "gamma": [0, 0.1, 0.3],
    },
    "lgbm": {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [-1, 3, 5, 7, 10, 20],
        "learning_rate": [0.05, 0.1, 0.2],
        "reg_lambda": [0.0, 1.0, 5.0],
        "reg_alpha": [0.0, 2.0, 5.0],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_samples": [10, 20, 30],
    },
    "cb": {
        "iterations": [400, 600, 800, 1000, 1200],
        "learning_rate": [0.02, 0.03, 0.04],
        "depth": [3, 5, 6, 7, 10, 20],
        "l2_leaf_reg": [3, 5, 7, 10],
        "subsample": [0.8, 1.0],
        "border_count": [32, 64, 128],
        "random_strength": [0.5, 1.0, 2.0],
        "bagging_temperature": [0.5, 1.0, 2.0],
    },
}
