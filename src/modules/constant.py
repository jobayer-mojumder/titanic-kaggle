from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

MAX_FEATURES_NUM = 11

DECISION_TREE_BASE_SCORE = 0.78282
XGBOOST_BASE_SCORE = 0.81255
RANDOM_FOREST_BASE_SCORE = 0.81145
LIGHTGBM_BASE_SCORE = 0.81592
CATBOOST_BASE_SCORE = 0.83219

DECISION_TREE_KAGGLE_SCORE = 0.73205
XGBOOST_KAGGLE_SCORE = 0.74401
RANDOM_FOREST_KAGGLE_SCORE = 0.76076
LIGHTGBM_KAGGLE_SCORE = 0.75837
CATBOOST_KAGGLE_SCORE = 0.7799

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
        use_label_encoder=False,
        eval_metric="logloss",
    ),
    "lgbm": LGBMClassifier(random_state=42, verbose=-1),
    "cb": CatBoostClassifier(random_seed=42, verbose=0),
}

PARAM_GRIDS = {
    "dt": {
        "criterion": ["gini", "entropy"],
        "splitter": ["best"],
        "max_depth": [2, 3, 5],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "rf": {
        "n_estimators": [50, 100, 200],
        "max_depth": [2, 3, 5],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True],
    },
    "xgb": {
        "n_estimators": [50, 100, 200],
        "max_depth": [2, 3, 5],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [7, 15, 31],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "lgbm": {
        "n_estimators": [50, 100, 200],
        "max_depth": [2, 3, 5],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [7, 15, 31],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "cb": {
        "iterations": [50, 100, 200],
        "depth": [2, 3, 5],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "l2_leaf_reg": [1, 3, 5],
        "border_count": [32, 64, 128],
    },
}
