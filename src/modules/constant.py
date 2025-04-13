from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

DECISION_TREE_BASE_SCORE = 0.80920
XGBOOST_BASE_SCORE = 0.82043
RANDOM_FOREST_BASE_SCORE = 0.79136
LIGHTGBM_BASE_SCORE = 0.82382
CATBOOST_BASE_SCORE = 0.81933

BASELINE_SCORE = {
    "dt": DECISION_TREE_BASE_SCORE,
    "xgb": XGBOOST_BASE_SCORE,
    "rf": RANDOM_FOREST_BASE_SCORE,
    "lgbm": LIGHTGBM_BASE_SCORE,
    "cb": CATBOOST_BASE_SCORE,
}

DECISION_TREE_KAGGLE_SCORE = 0.77511
XGBOOST_KAGGLE_SCORE = 0.78468
RANDOM_FOREST_KAGGLE_SCORE = 0.77751
LIGHTGBM_KAGGLE_SCORE = 0.76794
CATBOOST_KAGGLE_SCORE = 0.78708

KAGGLE_BASELINE_SCORE = {
    "dt": DECISION_TREE_KAGGLE_SCORE,
    "xgb": XGBOOST_KAGGLE_SCORE,
    "rf": RANDOM_FOREST_KAGGLE_SCORE,
    "lgbm": LIGHTGBM_KAGGLE_SCORE,
    "cb": CATBOOST_KAGGLE_SCORE,
}

# Default aligned configurations for fair comparison
DEFAULT_MODELS = {
    "dt": DecisionTreeClassifier(max_depth=3, random_state=42),
    "rf": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
    "xgb": XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    ),
    "lgbm": LGBMClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1
    ),
    "cb": CatBoostClassifier(
        iterations=100, depth=3, learning_rate=0.1, random_seed=42, verbose=0
    ),
}
