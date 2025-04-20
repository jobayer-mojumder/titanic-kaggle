import os
import pandas as pd

MODEL_INDEX = {
    "dt": 1,
    "xgb": 2,
    "rf": 3,
    "lgbm": 4,
    "cb": 5,
}


def load_combination_file(model_key, mode="kaggle"):
    base_path = f"results/{mode}/features-combinations"
    model_index = MODEL_INDEX.get(model_key)
    if not model_index:
        raise ValueError(f"Invalid model key: {model_key}")

    filename = f"{model_index}_{model_key}_comb.csv"
    file_path = os.path.join(base_path, filename)

    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return None

    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Failed to read {file_path}: {e}")
        return None


def get_best_single_feature_combination(model_key, mode="kaggle"):
    df = load_combination_file(model_key, mode=mode)
    if (
        df is None
        or ("kaggle_score" not in df.columns and "accuracy" not in df.columns)
        or "feature_num" not in df.columns
    ):
        return []

    score_col = "kaggle_score" if mode == "kaggle" else "accuracy"
    best_row = df.sort_values(by=score_col, ascending=False).iloc[0]
    feature_str = best_row["feature_num"]

    try:
        return [int(x.strip()) for x in feature_str.split(",")]
    except Exception as e:
        print(f"⚠️ Error parsing feature combination '{feature_str}': {e}")
        return []


def get_10_best_feature_combinations(model_key, mode="kaggle"):
    df = load_combination_file(model_key, mode=mode)
    if (
        df is None
        or ("kaggle_score" not in df.columns and "accuracy" not in df.columns)
        or "feature_num" not in df.columns
    ):
        return []

    score_col = "kaggle_score" if mode == "kaggle" else "accuracy"
    df_sorted = df.sort_values(by=score_col, ascending=False).drop_duplicates(
        subset="feature_num"
    )
    top_10 = df_sorted.head(10)
    combinations = []

    for _, row in top_10.iterrows():
        try:
            combo = [int(x.strip()) for x in row["feature_num"].split(",")]
            combinations.append(combo)
        except Exception as e:
            print(f"⚠️ Skipping invalid row: {row['feature_num']} ({e})")

    return combinations


def get_10_balanced_feature_combinations(model_key, mode="kaggle"):
    df = load_combination_file(model_key, mode=mode)
    if (
        df is None
        or ("kaggle_score" not in df.columns and "accuracy" not in df.columns)
        or "feature_num" not in df.columns
    ):
        return []

    score_col = "kaggle_score" if mode == "kaggle" else "accuracy"
    df_sorted = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    total = len(df_sorted)

    if total < 10:
        print(f"⚠️ Not enough data for balanced selection in model {model_key}")
        return get_10_best_feature_combinations(model_key, mode=mode)

    top_rows = df_sorted.head(3)
    mid_percentiles = [0.4, 0.5, 0.6, 0.7]
    mid_indices = sorted(set(int(total * p) for p in mid_percentiles))
    mid_rows = df_sorted.iloc[mid_indices]
    bottom_rows = df_sorted.tail(3)

    balanced_rows = pd.concat([top_rows, mid_rows, bottom_rows])
    combinations = []

    for _, row in balanced_rows.iterrows():
        try:
            combo = [int(x.strip()) for x in row["feature_num"].split(",")]
            combinations.append(combo)
        except Exception as e:
            print(f"⚠️ Skipping invalid row: {row['feature_num']} ({e})")

    return combinations
