# type: ignore
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from modules.constant import KAGGLE_BASELINE_SCORE, BASELINE_SCORE

SUMMARY_FILE = "results/results_summary.csv"
KAGGLE_FILE = "data/perfect_submission.csv"

model_index_map = {
    "dt": 1,
    "xgb": 2,
    "rf": 3,
    "lgbm": 4,
    "cb": 5,
}


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


def log_results(
    model_name, feature_list, accuracy, submission_file=None, tuned=False, std=None
):
    os.makedirs("results", exist_ok=True)

    improvement = accuracy - BASELINE_SCORE.get(model_name, 0)
    row = {
        "model": model_name,
        "features": ", ".join(feature_list) if feature_list else "baseline",
        "accuracy": truncate_float(accuracy),
        "std": truncate_float(std) if std is not None else None,
        "improvement": truncate_float(improvement),
        "tuned": tuned,
    }

    if os.path.exists(SUMMARY_FILE) and os.path.getsize(SUMMARY_FILE) > 0:
        df = pd.read_csv(SUMMARY_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(SUMMARY_FILE, index=False)

    if submission_file:
        compare_with_kaggle(submission_file, model_name, feature_list, tuned=tuned)


def compare_with_kaggle(submission_file, model_name, features, tuned=False):
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

    results_dir = "results/kaggle"
    os.makedirs(results_dir, exist_ok=True)
    index = model_index_map.get(model_name, 0)
    output_file = os.path.join(results_dir, f"{index}_{model_name}_kaggle_results.csv")

    row = {
        "model": model_name,
        "features": ", ".join(features) if features else "baseline",
        "accuracy_vs_kaggle": acc,
        "improvement": truncate_float(improvement),
        "tuned": tuned,
    }

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        df = pd.read_csv(output_file)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(output_file, index=False)
    return acc
