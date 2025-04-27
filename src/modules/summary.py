import os
import pandas as pd
from sklearn.metrics import accuracy_score
from modules.constant import KAGGLE_BASELINE_SCORE, BASELINE_SCORE
from modules.feature_implementation import FEATURE_MAP

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


def is_result_duplicate(file_path, model_name, feature_str, tuned_flag):
    if not os.path.exists(file_path):
        return False
    try:
        df = pd.read_csv(file_path, dtype={"feature_num": str})
        df["model"] = df["model"].astype(str).str.strip()
        df["feature_num"] = df["feature_num"].astype(str).str.strip()
        df["tuned"] = df["tuned"].astype(int)
    except Exception:
        return False

    return not df[
        (df["model"] == model_name)
        & (df["feature_num"] == feature_str)
        & (df["tuned"] == tuned_flag)
    ].empty


def result_already_logged(model_name, feature_num, tuned=False):
    feature_num = sorted(feature_num)
    feature_str = ", ".join(map(str, feature_num)) if feature_num else "baseline"
    tuned_flag = 1 if tuned else 0

    for kaggle in [False, True]:
        base_dir = "results/kaggle" if kaggle else "results/local"
        file_path = get_result_path(base_dir, model_name, feature_num, tuned)
        if is_result_duplicate(file_path, model_name, feature_str, tuned_flag):
            return True
    return False


def update_summary_csv(mode, row):
    assert mode in ["local", "kaggle"]
    summary_dir = "results"
    os.makedirs(summary_dir, exist_ok=True)
    file_path = os.path.join(summary_dir, f"summary_{mode}.csv")

    summary_row = {
        "model": row.get("model"),
        "tuned": row.get("tuned", 0),
        "feature_num": row.get("feature_num", "baseline"),
        "tuning_params": row.get("params", None),
        "improvement": row.get("improvement"),
        "baseline": row.get("baseline"),
    }

    if mode == "local":
        summary_row["accuracy"] = row.get("accuracy")
        summary_row["cv_std"] = row.get("std")
        summary_row["cv_scores"] = row.get("cv_scores")
    else:
        summary_row["kaggle_score"] = row.get("kaggle_score")

    column_order = [
        "model",
        "tuned",
        "feature_num",
        "baseline",
        "accuracy" if mode == "local" else "kaggle_score",
        "improvement",
        "cv_std" if mode == "local" else None,
        "tuning_params",
        "cv_scores" if mode == "local" else None,
    ]
    column_order = [col for col in column_order if col is not None]
    summary_row = {col: summary_row.get(col) for col in column_order}

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=column_order)

    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    df.to_csv(file_path, index=False)
    print(f"📄 Summary Updated for {mode}")


def log_results(
    model_name,
    feature_list,
    accuracy,
    submission_file=None,
    tuned=False,
    params=None,
    std=None,
    cv_scores=None,
):
    feature_list = sorted(normalize_feature_list(feature_list))
    feature_str = ", ".join(map(str, feature_list)) if feature_list else "baseline"

    if model_name in BASELINE_SCORE:
        compare_with_baseline(
            accuracy, BASELINE_SCORE[model_name], label="Local Accuracy"
        )

    improvement = accuracy - BASELINE_SCORE.get(model_name, 0)

    row = {
        "model": model_name,
        "tuned": 1 if tuned else 0,
        "feature_num": feature_str,
        "baseline": BASELINE_SCORE.get(model_name, 0),
        "accuracy": truncate_float(accuracy),
        "std": truncate_float(std) if std is not None else None,
        "improvement": truncate_float(improvement),
        "params": str(params) if params else None,
        "cv_scores": str(cv_scores) if cv_scores is not None else None,
    }

    update_summary_csv("local", row)
    local_file = get_result_path("results/local", model_name, feature_list, tuned)

    if is_result_duplicate(local_file, model_name, feature_str, row["tuned"]):
        print(
            f"⏭️ Local result already logged for {model_name} with features [{feature_str}] (tuned={row['tuned']})"
        )
        return

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
    features = sorted(normalize_feature_list(features))
    feature_str = ", ".join(map(str, features)) if features else "baseline"

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
        "feature_num": feature_str,
        "baseline": KAGGLE_BASELINE_SCORE.get(model_name, 0),
        "kaggle_score": acc,
        "improvement": truncate_float(improvement),
        "tuned": 1 if tuned else 0,
        "params": str(params) if params else None,
    }

    update_summary_csv("kaggle", row)
    kaggle_file = get_result_path("results/kaggle", model_name, features, tuned)

    if is_result_duplicate(kaggle_file, model_name, feature_str, row["tuned"]):
        print(
            f"⏭️ Kaggle result already logged for {model_name} with features [{feature_str}] (tuned={row['tuned']})"
        )
        return acc

    if os.path.exists(kaggle_file) and os.path.getsize(kaggle_file) > 0:
        df = pd.read_csv(kaggle_file)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(kaggle_file, index=False)
    print(f"📁 Appended Kaggle results to {kaggle_file}")
    return acc


def get_submission_path(model_key, feature_num):
    base_dir = os.getcwd()
    index = model_index_map.get(model_key, 0)
    folder_name = f"{index}_{model_key}"
    sorted_features = sorted(feature_num)
    suffix = "_".join(map(str, sorted_features)) if sorted_features else "base"
    out_dir = os.path.join(base_dir, "submissions", folder_name)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"submission_{model_key}_{suffix}.csv"
    return os.path.join(out_dir, filename)


def save_feature_importance(model, preproc, model_key, feature_num, tuned=False):
    if not hasattr(model, "feature_importances_"):
        print("⚠️ Model does not support feature_importances_. Skipping.")
        return

    # Get feature names
    try:
        feature_names = preproc.get_feature_names_out()
    except AttributeError:
        feature_names = [
            f"feature_{i}" for i in range(model.feature_importances_.shape[0])
        ]

    importance = model.feature_importances_

    df = pd.DataFrame(
        {
            "model_key": model_key,
            "feature_num": (
                ", ".join(map(str, sorted(feature_num))) if feature_num else "baseline"
            ),
            "tuned": int(tuned),
            "feature": feature_names,
            "importance": importance,
        }
    )

    # Sort by importance
    df = df.sort_values(by="importance", ascending=False)

    # Create directory
    out_path = "results"
    os.makedirs(out_path, exist_ok=True)

    file_path = os.path.join(out_path, "feature_importance.csv")

    # Save (append mode)
    if os.path.exists(file_path):
        existing = pd.read_csv(file_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(file_path, index=False)
    print(f"✅ Saved feature importance to {file_path}")
