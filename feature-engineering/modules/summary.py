# type: ignore
import pandas as pd
from sklearn.metrics import accuracy_score
import os

from modules.constant import KAGGLE_BASELINE_SCORE, BASELINE_SCORE


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


def log_results(model_name, feature_list, accuracy, submission_file=None):
    local_output_file = "results_summary.csv"

    improvement = accuracy - BASELINE_SCORE[model_name]

    row = {
        "model": model_name,
        "features": ", ".join(feature_list) if feature_list else "baseline",
        "accuracy": accuracy,
        "improvement": truncate_float(improvement),
    }

    if os.path.exists(local_output_file) and os.path.getsize(local_output_file) > 0:
        df = pd.read_csv(local_output_file)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(local_output_file, index=False)

    if submission_file:
        compare_with_kaggle(submission_file, model_name, feature_list)


def compare_with_kaggle(submission_file, model_name, features):
    kaggle_file = "../perfect_submission.csv"
    output_file = "kaggle_results.csv"

    if not os.path.exists(kaggle_file):
        print("⚠️ Kaggle perfect submission not found. Skipping comparison.")
        return None

    user_df = pd.read_csv(submission_file).set_index("PassengerId")
    kaggle_df = pd.read_csv(kaggle_file).set_index("PassengerId")

    if not user_df.index.equals(kaggle_df.index):
        print("❌ PassengerId mismatch, cannot compare.")
        return None

    acc = accuracy_score(kaggle_df["Survived"], user_df["Survived"])
    acc = truncate_float(acc)

    print(f"📊 Accuracy vs Kaggle perfect: \033[94m{acc}\033[0m")

    if model_name in KAGGLE_BASELINE_SCORE:
        compare_with_baseline(
            acc, KAGGLE_BASELINE_SCORE[model_name], label="Kaggle Accuracy"
        )

    improvement = acc - KAGGLE_BASELINE_SCORE[model_name]
    # Log to CSV
    row = {
        "model": model_name,
        "features": ", ".join(features) if features else "baseline",
        "accuracy_vs_kaggle": acc,
        "improvement": truncate_float(improvement),
    }

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        df = pd.read_csv(output_file)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(output_file, index=False)
    return acc
