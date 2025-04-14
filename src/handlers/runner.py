from handlers.utils import prompt_all_or_one, select_model_key
import time
import os
import pandas as pd


def load_finished_combinations(model_key):
    model_index_map = {"dt": 1, "xgb": 2, "rf": 3, "lgbm": 4, "cb": 5}
    index = model_index_map[model_key]
    path = f"results/kaggle/tuning-combinations/{index}_{model_key}_comb_tuned.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        return set(df["feature_nums"].dropna().astype(str))
    return set()


def run_all_single_features(run_model_func, model_key, tune=False):
    for feature_num in range(1, 12):
        run_model_func(model_key, [feature_num], tune=tune)


def run_all_general_combinations(run_model_func, model_key, tune=False):
    from modules.combination import GENERAL_FEATURE_COMBINATIONS

    finished_combos = set()
    if tune:
        finished_combos = load_finished_combinations(model_key)

    for combo in GENERAL_FEATURE_COMBINATIONS:
        feature_nums_str = ", ".join(map(str, combo))
        if tune and feature_nums_str in finished_combos:
            print(f"⏭️  Skipping already tuned combo: {feature_nums_str}")
            continue

        run_model_func(model_key, combo, tune=tune)


def run_model_combinations(run_model_func, model_key, tune=False):
    from modules.combination import (
        DT_COMBINATIONS,
        XGB_COMBINATIONS,
        RF_COMBINATIONS,
        LGBM_COMBINATIONS,
        CB_COMBINATIONS,
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


def run_baseline_models(run_model_func, model_key, tune=False):
    run_model_func(model_key, [], tune=tune)
