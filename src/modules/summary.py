import os
import pandas as pd
from sklearn.metrics import accuracy_score
from modules.constant import KAGGLE_BASELINE_SCORE, BASELINE_SCORE
from modules.feature_implementation import FEATURE_MAP

# Mapping model name to index
model_index_map = {
    "dt": 1,
    "xgb": 2,
    "rf": 3,
    "lgbm": 4,
    "cb": 5,
}

FEATURE_NAME_TO_NUM = {v.lower(): k for k, v in FEATURE_MAP.items()}

KAGGLE_FILE = "data/perfect_submission.csv"


def truncate_float(value, digits=5):
    factor = 10.0**digits
    return int(value * factor) / factor


def compare_with_baseline(score, baseline, label="Accuracy"):
    score = round(score, 5)
    baseline = round(baseline, 5)

    if score == baseline:
        print(f"\033[93m⚠️  {label} matches baseline: {score} \033[0m")
    elif score > baseline:
        print(
            f"\033[92m✅ {label} improved: {score} vs baseline {baseline} = +{(score - baseline):.5f} \033[0m"
        )
    else:
        print(
            f"\033[91m⚠️  {label} dropped: {score} vs baseline {baseline} = -{(baseline - score):.5f} \033[0m"
        )


def normalize_feature_list(feature_list):
    if not feature_list:
        return []
    normalized = []
    for f in feature_list:
        try:
            normalized.append(int(f))
        except ValueError:
            f_lower = str(f).lower()
            if f_lower in FEATURE_NAME_TO_NUM:
                normalized.append(FEATURE_NAME_TO_NUM[f_lower])
            else:
                print(f"⚠️ Unknown feature: {f}, skipping")
    return normalized


def get_feature_names(feature_list):
    if not feature_list:
        return "baseline"
    return ", ".join(FEATURE_MAP.get(int(f), f"F{f}") for f in feature_list)


def get_result_path(base_dir, model_name, feature_list, tuned=False):
    model_index = model_index_map.get(model_name, 0)
    feature_count = len(feature_list) if feature_list else 0

    if feature_count == 1:
        folder = "single-tuning" if tuned else "single-features"
        filename = (
            f"{model_index}_{model_name}_single_tuned.csv"
            if tuned
            else f"{model_index}_{model_name}_single.csv"
        )
    elif feature_count > 1:
        folder = "tuning-combinations" if tuned else "features-combinations"
        filename = (
            f"{model_index}_{model_name}_comb_tuned.csv"
            if tuned
            else f"{model_index}_{model_name}_comb.csv"
        )
    else:
        folder = "baseline"
        filename = f"{model_index}_{model_name}_baseline.csv"

    full_dir = os.path.join(base_dir, folder)
    os.makedirs(full_dir, exist_ok=True)
    return os.path.join(full_dir, filename)


def log_results(
    model_name,
    feature_list,
    accuracy,
    submission_file=None,
    tuned=False,
    params=None,
    std=None,
):
    feature_list = normalize_feature_list(feature_list)
    improvement = accuracy - BASELINE_SCORE.get(model_name, 0)

    row = {
        "model": model_name,
        "feature_nums": (
            ", ".join(map(str, feature_list)) if feature_list else "baseline"
        ),
        "features": get_feature_names(feature_list),
        "baseline": BASELINE_SCORE.get(model_name, 0),
        "accuracy": truncate_float(accuracy),
        "std": truncate_float(std) if std is not None else None,
        "improvement": truncate_float(improvement),
        "tuned": tuned,
        "params": str(params) if params else None,
    }

    local_file = get_result_path("results/local", model_name, feature_list, tuned)

    if os.path.exists(local_file) and os.path.getsize(local_file) > 0:
        df = pd.read_csv(local_file)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(local_file, index=False)
    print(f"📁 Appended local results to {local_file}")

    if submission_file:
        compare_with_kaggle(submission_file, model_name, feature_list, tuned, params)


def compare_with_kaggle(
    submission_file, model_name, features, tuned=False, params=None
):
    features = normalize_feature_list(features)

    if not os.path.exists(KAGGLE_FILE):
        print("⚠️ Kaggle perfect submission not found. Skipping comparison.")
        return None

    user_df = pd.read_csv(submission_file).set_index("PassengerId")
    kaggle_df = pd.read_csv(KAGGLE_FILE).set_index("PassengerId")

    if not user_df.index.equals(kaggle_df.index):
        print("❌ PassengerId mismatch, cannot compare.")
        return None

    acc = truncate_float(accuracy_score(kaggle_df["Survived"], user_df["Survived"]))
    print(f"📊 Accuracy vs Kaggle perfect: \033[94m{acc}\033[0m")

    if model_name in KAGGLE_BASELINE_SCORE:
        compare_with_baseline(
            acc, KAGGLE_BASELINE_SCORE[model_name], label="Kaggle Accuracy"
        )

    improvement = acc - KAGGLE_BASELINE_SCORE.get(model_name, 0)

    row = {
        "model": model_name,
        "feature_nums": ", ".join(map(str, features)) if features else "baseline",
        "features": get_feature_names(features),
        "baseline": KAGGLE_BASELINE_SCORE.get(model_name, 0),
        "accuracy_vs_kaggle": acc,
        "improvement": truncate_float(improvement),
        "tuned": tuned,
        "params": str(params) if params else None,
    }

    kaggle_file = get_result_path("results/kaggle", model_name, features, tuned)

    if os.path.exists(kaggle_file) and os.path.getsize(kaggle_file) > 0:
        df = pd.read_csv(kaggle_file)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(kaggle_file, index=False)
    print(f"📁 Appended Kaggle results to {kaggle_file}")
    return acc


def get_submission_path(model_key, feature_nums):
    base_dir = os.getcwd()
    index = model_index_map.get(model_key, 0)
    folder_name = f"{index}_{model_key}"

    sorted_features = sorted(feature_nums)
    suffix = "_".join(map(str, sorted_features)) if sorted_features else "base"

    out_dir = os.path.join(base_dir, "submissions", folder_name)
    os.makedirs(out_dir, exist_ok=True)

    filename = f"submission_{model_key}_{suffix}.csv"
    return os.path.join(out_dir, filename)
