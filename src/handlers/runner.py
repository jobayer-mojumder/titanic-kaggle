from handlers.utils import prompt_all_or_one, select_model_key
import time

def run_all_single_features(run_model_func, model_key, tune=False):
    for feature_num in range(1, 12):
        run_model_func(model_key, [feature_num], tune=tune)
    run_model_func(model_key, [], tune=tune)

def run_all_general_combinations(run_model_func, model_key, tune=False):
    from modules.combination import GENERAL_FEATURE_COMBINATIONS
    for combo in GENERAL_FEATURE_COMBINATIONS:
        run_model_func(model_key, combo, tune=tune)

def run_best_combinations(run_model_func, model_key, tune=False):
    from modules.combination import (
        DT_COMBINATIONS, XGB_COMBINATIONS, RF_COMBINATIONS,
        LGBM_COMBINATIONS, CB_COMBINATIONS
    )
    combos = {
        "dt": DT_COMBINATIONS,
        "xgb": XGB_COMBINATIONS,
        "rf": RF_COMBINATIONS,
        "lgbm": LGBM_COMBINATIONS,
        "cb": CB_COMBINATIONS,
    }
    for combo in combos.get(model_key, []):
        run_model_func(model_key, combo, tune=tune)

def run_all_models(run_model_func, runner_fn, tune=False):
    for model_key in ["dt", "xgb", "rf", "lgbm", "cb"]:
        runner_fn(run_model_func, model_key, tune=tune)