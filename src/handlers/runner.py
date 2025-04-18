import os
import pandas as pd
from modules.analysis import (
    get_best_single_feature_combination,
    get_10_best_feature_combinations,
    get_10_balanced_feature_combinations,
)


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
            print(f"⏭️ {model_key} - Skipping already tuned combo: {feature_nums_str}")
            continue

        run_model_func(model_key, combo, tune=tune)


def run_all_models(run_model_func, runner_fn, tune=False):
    for model_key in ["dt", "xgb", "rf", "lgbm", "cb"]:
        runner_fn(run_model_func, model_key, tune=tune)


def run_baseline_models(run_model_func, model_key, tune=False):
    run_model_func(model_key, [], tune=tune)


def run_all_features_in_one_combination(run_model_func, model_key, tune=False):
    from modules.combination import ALL_FEATURE_COMBINATION

    finished_combos = set()
    if tune:
        finished_combos = load_finished_combinations(model_key)
    feature_nums_str = ", ".join(map(str, ALL_FEATURE_COMBINATION))
    if tune and feature_nums_str in finished_combos:
        print(f"⏭️ {model_key} - Skipping already tuned combo: {feature_nums_str}")
        return
    run_model_func(model_key, ALL_FEATURE_COMBINATION, tune=tune)


def run_best_single_feature_combination(run_model_func, model_key, tune=False):
    combination = get_best_single_feature_combination(model_key)

    if not combination:
        print(f"⚠️ No best single feature combination found for {model_key}.")
        return

    finished_combos = set()
    if tune:
        finished_combos = load_finished_combinations(model_key)
    feature_nums_str = ", ".join(map(str, combination))
    if tune and feature_nums_str in finished_combos:
        print(f"⏭️ {model_key} - Skipping already tuned combo: {feature_nums_str}")
        return
    run_model_func(model_key, combination, tune=tune)


def run_10_balanced_feature_combinations(run_model_func, model_key, tune=False):
    combinations = get_10_balanced_feature_combinations(model_key)

    if not combinations:
        print(f"⚠️ No 10 balanced feature combinations found for {model_key}.")
        return

    finished_combos = set()
    if tune:
        finished_combos = load_finished_combinations(model_key)

    for combo in combinations:
        feature_nums_str = ", ".join(map(str, combo))
        if tune and feature_nums_str in finished_combos:
            print(f"⏭️ {model_key} - Skipping already tuned combo: {feature_nums_str}")
            continue

        run_model_func(model_key, combo, tune=tune)


def run_10_best_feature_combinations(run_model_func, model_key, tune=False):
    combinations = get_10_best_feature_combinations(model_key)

    if not combinations:
        print(f"⚠️ No 10 best feature combinations found for {model_key}.")
        return

    finished_combos = set()
    if tune:
        finished_combos = load_finished_combinations(model_key)

    for combo in combinations:
        feature_nums_str = ", ".join(map(str, combo))
        if tune and feature_nums_str in finished_combos:
            print(f"⏭️ {model_key} - Skipping already tuned combo: {feature_nums_str}")
            continue

        run_model_func(model_key, combo, tune=tune)
