import os
import pandas as pd
from tabulate import tabulate
from modules.analysis import (
    get_best_single_feature_combination,
    get_10_best_feature_combinations,
    get_10_balanced_feature_combinations,
)
from modules.constant import KAGGLE_BASELINE_SCORE

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_DIR = os.path.join(CURRENT_DIR, "stats/kaggle")
os.makedirs(STATS_DIR, exist_ok=True)

MODEL_KEYS = {
    "1": ("dt", "Decision Tree", 1),
    "2": ("xgb", "XGBoost", 2),
    "3": ("rf", "Random Forest", 3),
    "4": ("lgbm", "LightGBM", 4),
    "5": ("cb", "CatBoost", 5),
}
MODEL_KEYS_REV = {v[0]: k for k, v in MODEL_KEYS.items()}


def select_model():
    print("\n🔧 Choose a model:")
    for k, (_, name, _) in MODEL_KEYS.items():
        print(f"{k}. {name}")
    choice = input("Enter model number: ").strip()
    return MODEL_KEYS.get(choice, (None, None, None))


def display_table(df, title, file_name):
    os.system("cls" if os.name == "nt" else "clear")

    if df.empty:
        print("⚠️ No data found.")
        return

    print(f"\n📊 {title}")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))

    output_path = os.path.join(STATS_DIR, file_name)
    df.to_csv(output_path, index=False)
    print(f"\n📁 Data saved to stats/{os.path.basename(output_path)}")

    input("Press Enter to continue...")


def show_single_feature_results(model_key, model_label, model_index, tuned=False):
    subdir = "single-tuning" if tuned else "single-features"
    suffix = "single_tuned.csv" if tuned else "single.csv"
    filename = f"{model_index}_{model_key}_{suffix}"
    path = os.path.join("results", "kaggle", subdir, filename)
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("⚠️ No data found.")
        return
    df["rank"] = df["kaggle_score"].rank(method="min", ascending=False).astype(int)
    df = df.sort_values(by="kaggle_score", ascending=False)
    df = df[["rank", "feature_nums", "kaggle_score", "improvement", "tuned"]]
    title = (
        f"Tuned Single Feature Results for {model_label}"
        if tuned
        else f"Single Feature Results for {model_label}"
    )
    out_csv = (
        f"{model_index}_{model_key}_single_tuned.csv"
        if tuned
        else f"{model_index}_{model_key}_single_features.csv"
    )
    display_table(df, title, out_csv)


def show_baseline_scores(tuned=False):
    summary_path = os.path.join("results", "summary_kaggle.csv")
    if not os.path.exists(summary_path):
        print(f"❌ summary_kaggle.csv not found at {summary_path}")
        return

    df = pd.read_csv(summary_path)
    if tuned:
        filtered = df[(df["feature_nums"] == "baseline") & (df["tuned"] == 1)]
        label = "Tuned"
    else:
        filtered = df[(df["feature_nums"] == "baseline") & (df["tuned"] == 0)]
        label = "Untuned"

    rows = []
    for _, (model_key, model_label, _) in MODEL_KEYS.items():
        row = filtered[filtered["model"] == model_key]
        if not row.empty:
            score = row.iloc[0]["kaggle_score"]
            rows.append((model_label, model_key, round(score, 5)))
        else:
            rows.append((model_label, model_key, "Not found"))

    out_df = pd.DataFrame(
        rows, columns=["Model", "Key", f"{label} Baseline Kaggle Score"]
    )
    filename = f"baseline_scores_{'tuned' if tuned else 'features'}.csv"
    display_table(out_df, f"Baseline Kaggle Scores ({label}) for All Models", filename)


def load_tuned_combination_file(model_key, model_index):
    path = os.path.join(
        "results",
        "kaggle",
        "tuning-combinations",
        f"{model_index}_{model_key}_comb_tuned.csv",
    )
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        return None
    return pd.read_csv(path)


def lookup_combinations_in_tuned(tuned_df, combinations):
    tuned_df = tuned_df.copy()
    tuned_df["feature_nums"] = tuned_df["feature_nums"].astype(str).str.strip()
    combo_strs = [", ".join(map(str, sorted(combo))) for combo in combinations]
    return tuned_df[tuned_df["feature_nums"].isin(combo_strs)]


def extract_rows_for_combos(model_key, combos, tuned=False):
    folder = "tuning-combinations" if tuned else "features-combinations"
    suffix = "_comb_tuned.csv" if tuned else "_comb.csv"
    index = MODEL_KEYS_REV[model_key]
    path = f"results/kaggle/{folder}/{index}_{model_key}{suffix}"
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    df["feature_nums"] = df["feature_nums"].astype(str).str.strip()
    rows = []
    for combo in combos:
        combo_str = ", ".join(map(str, sorted(combo)))
        match = df[df["feature_nums"] == combo_str]
        if not match.empty:
            rows.append(match.iloc[0].to_dict())
    return rows


def show_best_combinations_all(tuned=False):
    rows = []
    for _, (model_key, model_label, model_index) in MODEL_KEYS.items():
        combo = get_best_single_feature_combination(model_key)
        result = extract_rows_for_combos(model_key, [combo], tuned=tuned)
        if result:
            row = result[0]
            rows.append(
                {
                    "rank": model_index,
                    "model": model_label,
                    "model_key": model_key,
                    "feature_nums": row["feature_nums"],
                    "kaggle_score": row["kaggle_score"],
                    "improvement": row.get("improvement", ""),
                    "tuned": row.get("tuned", ""),
                }
            )
    df = pd.DataFrame(rows).sort_values(by="rank").reset_index(drop=True)
    title = (
        "Best Tuned Feature Combination for All Models"
        if tuned
        else "Best Feature Combination for All Models"
    )
    file = "best_combinations_tuned.csv" if tuned else "best_combinations_features.csv"
    display_table(df, title, file)


def show_combinations(
    model_key, model_label, model_index, combos, title, filename, tuned=False
):
    rows = extract_rows_for_combos(model_key, combos, tuned=tuned)
    if not rows:
        print("⚠️ No matching rows found.")
        return
    df = pd.DataFrame(rows)
    df.insert(0, "rank", range(1, len(df) + 1))
    df = df[["rank", "feature_nums", "kaggle_score", "improvement", "tuned"]]
    display_table(df, title, filename)


def print_menu():
    os.system("cls" if os.name == "nt" else "clear")
    print("\n" + "=" * 50)
    print("🎯 Stats Menu".center(50))
    print("=" * 50)

    print("\n📊 Feature Engineering")
    print("     [1]  Best feature combination for all models")
    print("     [2]  Top 10 feature combinations for a model")
    print("     [3]  Balanced 10 feature combinations for a model")
    print("     [4]  Single feature results for a model")
    print("     [5]  Baseline Kaggle score (untuned) for all models")

    print("\n🔧 Model Tuning (Single Features)")
    print("     [6]  Tuned single feature results for a model")
    print("     [7]  Tuned baseline score for all models")

    print("\n🧪 Effect of Model Tuning on Engineered Feature Combinations")
    print("     [8]  Tuned score for best feature combination (from FE)")
    print("     [9]  Tuned scores for top 10 feature combinations (from FE)")
    print("     [10]  Tuned scores for balanced 10 feature combinations (from FE)")

    print("\n  [0]  Exit")
    print("=" * 50)


def stats_menu():
    while True:
        print_menu()
        choice = input("Choose an option: ").strip()
        if choice == "0":
            break
        elif choice == "1":
            show_best_combinations_all(tuned=False)
        elif choice == "2":
            model_key, model_label, model_index = select_model()
            if model_key:
                combos = get_10_best_feature_combinations(model_key)
                show_combinations(
                    model_key,
                    model_label,
                    model_index,
                    combos,
                    f"Top 10 Feature Combinations for {model_label}",
                    f"{model_index}_{model_key}_top10.csv",
                    tuned=False,
                )
        elif choice == "3":
            model_key, model_label, model_index = select_model()
            if model_key:
                combos = get_10_balanced_feature_combinations(model_key)
                show_combinations(
                    model_key,
                    model_label,
                    model_index,
                    combos,
                    f"Balanced 10 Feature Combinations for {model_label}",
                    f"{model_index}_{model_key}_balanced10.csv",
                    tuned=False,
                )
        elif choice == "4":
            model_key, model_label, model_index = select_model()
            if model_key:
                show_single_feature_results(
                    model_key, model_label, model_index, tuned=False
                )
        elif choice == "5":
            show_baseline_scores(tuned=False)
        elif choice == "6":
            model_key, model_label, model_index = select_model()
            if model_key:
                show_single_feature_results(
                    model_key, model_label, model_index, tuned=True
                )
        elif choice == "7":
            show_baseline_scores(tuned=True)
        elif choice == "8":
            show_best_combinations_all(tuned=True)
        elif choice == "9":
            model_key, model_label, model_index = select_model()
            if model_key:
                combos = get_10_best_feature_combinations(model_key)
                show_combinations(
                    model_key,
                    model_label,
                    model_index,
                    combos,
                    f"Top 10 Tuned Combinations for {model_label}",
                    f"{model_index}_{model_key}_top10_tuned.csv",
                    tuned=True,
                )
        elif choice == "10":
            model_key, model_label, model_index = select_model()
            if model_key:
                combos = get_10_balanced_feature_combinations(model_key)
                show_combinations(
                    model_key,
                    model_label,
                    model_index,
                    combos,
                    f"Balanced 10 Tuned Combinations for {model_label}",
                    f"{model_index}_{model_key}_balanced10_tuned.csv",
                    tuned=True,
                )
        else:
            print("❌ Invalid option.")


if __name__ == "__main__":
    stats_menu()
