from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

MAX_FEATURES_NUM = 11

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
    "rf": RANDOM_FOREST_BASE_SCORE,
    "xgb": XGBOOST_BASE_SCORE,
    "lgbm": LIGHTGBM_BASE_SCORE,
    "cb": CATBOOST_BASE_SCORE,
}

MODEL_ORDER = {
    "Decision Tree": 1,
    "Random Forest": 2,
    "XGBoost": 3,
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
        "criterion": ["gini", "entropy", "log_loss"],
        "splitter": ["random", "best"],
        "max_depth": [3, 5, 10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": [None, "sqrt", "log2"],
        "max_leaf_nodes": [10, 20, 50, 100, None],
        "class_weight": [None, "balanced"],
    },
    "rf": {
        "n_estimators": [50, 100, 200, 300],
        "criterion": ["entropy", "gini", "log_loss"],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", None],
        "max_leaf_nodes": [20, 50, None],
        "bootstrap": [False, True],
        "class_weight": [None, "balanced"],
    },
    "xgb": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.1, 0.2, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.3, 1.0],
        "reg_alpha": [0, 1, 5],
        "reg_lambda": [1, 5, 10],
        "scale_pos_weight": [1, 2, 5],
        "min_child_weight": [1, 3, 5],
        "tree_method": ["auto"],
        "booster": ["gbtree", "dart"],
    },
    "lgbm": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 10, -1],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [10, 20, 30],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 1, 5],
        "reg_lambda": [0, 1, 5],
        "boosting_type": ["gbdt", "dart"],
        "class_weight": [None, "balanced"],
    },
    "cb": {
        "iterations": [300, 400, 600, 800, 1000],
        "learning_rate": [0.01, 0.03, 0.05],
        "depth": [3, 4, 6, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 10],
        "border_count": [32, 64, 128],
        "random_strength": [0.5, 1, 2],
        "bagging_temperature": [0.5, 0.7, 1, 2],
        "grow_policy": ["SymmetricTree", "Depthwise", "Lossguide"],
        "bootstrap_type": ["Bayesian", "Bernoulli", "MVS"],
        "scale_pos_weight": [1, 2, 5],
    },
}
