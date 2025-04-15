import os
import pandas as pd
from tabulate import tabulate
from modules.analysis import (
    load_combination_file,
    get_best_single_feature_combination,
    get_10_best_feature_combinations,
    get_10_balanced_feature_combinations,
)
from modules.constant import KAGGLE_BASELINE_SCORE

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_DIR = os.path.join(CURRENT_DIR, "stats")
os.makedirs(STATS_DIR, exist_ok=True)

MODEL_KEYS = {
    "1": ("dt", "Decision Tree", 1),
    "2": ("xgb", "XGBoost", 2),
    "3": ("rf", "Random Forest", 3),
    "4": ("lgbm", "LightGBM", 4),
    "5": ("cb", "CatBoost", 5),
}


def extract_rows_for_combos(model_key, combos):
    df = load_combination_file(model_key)
    if df is None:
        return []
    df["feature_nums"] = df["feature_nums"].astype(str).str.strip()
    rows = []
    for combo in combos:
        combo_str = ", ".join(map(str, sorted(combo)))
        match = df[df["feature_nums"] == combo_str]
        if not match.empty:
            row = match.iloc[0].to_dict()
            row["feature_list"] = combo
            rows.append(row)
    return rows


def show_and_save(label, model_key, combos, filename_base, model_index):
    rows = extract_rows_for_combos(model_key, combos)
    if not rows:
        print(f"⚠️ No matching rows found.")
        return
    print(f"\n📊 {label}")
    df_out = pd.DataFrame(rows)
    df_out = df_out[["feature_nums", "kaggle_score", "improvement", "tuned"]]
    df_out.insert(0, "rank", range(1, len(df_out) + 1))
    print(tabulate(df_out, headers="keys", tablefmt="fancy_grid", showindex=False))
    filename = f"{model_index}_{model_key}_{filename_base}.csv"
    out_path = os.path.join(STATS_DIR, filename)
    df_out.to_csv(out_path, index=False)
    print(f"\n📁 Saved to {os.path.basename(out_path)}")


def select_model():
    print("\n🔧 Choose a model:")
    for k, (_, name, _) in MODEL_KEYS.items():
        print(f"{k}. {name}")
    model_choice = input("Enter model number: ").strip()
    return MODEL_KEYS.get(model_choice, (None, None, None))


def show_best_combinations_all():
    print("\n📊 Best Combination for All Models")
    all_rows = []
    for _, (model_key, model_label, model_index) in MODEL_KEYS.items():
        combo = get_best_single_feature_combination(model_key)
        if combo:
            rows = extract_rows_for_combos(model_key, [combo])
            if rows:
                row = rows[0]
                all_rows.append(
                    {
                        "rank": model_index,
                        "model": model_label,
                        "model_key": model_key,
                        "feature_nums": row["feature_nums"],
                        "kaggle_score": row["kaggle_score"],
                        "improvement": row["improvement"],
                        "tuned": row["tuned"],
                    }
                )
    df = pd.DataFrame(all_rows).sort_values(by="rank").reset_index(drop=True)
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
    out_path = os.path.join(STATS_DIR, "best_combinations_features.csv")
    df.to_csv(out_path, index=False)
    print(f"\n📁 Saved to {os.path.basename(out_path)}")


def show_top_10_combinations(model_key, model_label, model_index):
    combos = get_10_best_feature_combinations(model_key)
    show_and_save(
        f"Top 10 combinations for {model_label}",
        model_key,
        combos,
        "top10",
        model_index,
    )


def show_balanced_combinations(model_key, model_label, model_index):
    combos = get_10_balanced_feature_combinations(model_key)
    show_and_save(
        f"Balanced 10 combinations for {model_label}",
        model_key,
        combos,
        "balanced10",
        model_index,
    )


def show_single_feature_results(model_key, model_label, model_index):
    csv_path = os.path.join(
        "results", "kaggle", "single-features", f"{model_index}_{model_key}_single.csv"
    )
    if not os.path.exists(csv_path):
        print(f"⚠️ File not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("⚠️ No single-feature results found.")
        return
    df["rank"] = df["kaggle_score"].rank(method="min", ascending=False).astype(int)
    df = df.sort_values(by="kaggle_score", ascending=False)
    df = df[["rank", "feature_nums", "kaggle_score", "improvement", "tuned"]]
    print(f"\n📊 All 11 Single Feature Results for {model_label}")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
    filename = f"{model_index}_{model_key}_single_features.csv"
    out_path = os.path.join(STATS_DIR, filename)
    df.to_csv(out_path, index=False)
    print(f"\n📁 Saved to {os.path.basename(out_path)}")


def show_baseline_scores():
    print("\n📊 Baseline Kaggle Scores for All Models")
    data = []
    for _, (model_key, label, _) in MODEL_KEYS.items():
        score = KAGGLE_BASELINE_SCORE.get(model_key)
        data.append((label, model_key, score if score is not None else "N/A"))
    df = pd.DataFrame(data, columns=["Model", "Key", "Kaggle Score"])
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
    out_path = os.path.join(STATS_DIR, "baseline_scores_features.csv")
    df.to_csv(out_path, index=False)
    print(f"\n📁 Saved to {os.path.basename(out_path)}")


def show_single_tuned_results(model_key, model_label, model_index):
    csv_path = os.path.join(
        "results",
        "kaggle",
        "single-tuning",
        f"{model_index}_{model_key}_single_tuned.csv",
    )
    if not os.path.exists(csv_path):
        print(f"⚠️ File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("⚠️ No single-feature tuning results found.")
        return

    df["rank"] = df["kaggle_score"].rank(method="min", ascending=False).astype(int)
    df = df.sort_values(by="kaggle_score", ascending=False)
    df = df[["rank", "feature_nums", "kaggle_score", "improvement", "tuned"]]

    print(f"\n📊 Tuned Results for 11 Single Features — {model_label}")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))

    filename = f"{model_index}_{model_key}_single_tuned.csv"
    out_path = os.path.join(STATS_DIR, filename)
    df.to_csv(out_path, index=False)
    print(f"\n📁 Saved to {os.path.basename(out_path)}")


def show_baseline_tuned_scores():
    print("\n📊 Baseline Tuned Kaggle Scores for All Models")
    rows = []

    for _, (model_key, model_label, model_index) in MODEL_KEYS.items():
        file_path = os.path.join(
            "results",
            "kaggle",
            "single-tuning",
            f"{model_index}_{model_key}_single_tuned.csv",
        )

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if not df.empty and "kaggle_score" in df.columns:
                best_score = df["kaggle_score"].max()
                rows.append((model_label, model_key, round(best_score, 5)))
            else:
                rows.append((model_label, model_key, "N/A"))
        else:
            rows.append((model_label, model_key, "File Missing"))

    df = pd.DataFrame(rows, columns=["Model", "Key", "Tuned Baseline Kaggle Score"])
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))

    out_path = os.path.join(STATS_DIR, "baseline_scores_tuned.csv")
    df.to_csv(out_path, index=False)
    print(f"\n📁 Saved to {os.path.basename(out_path)}")


def stats_menu():
    while True:
        print("\n🎯 Stats Menu:")
        print("1. Show best combination for all models")
        print("2. Show 10 best combinations for a model")
        print("3. Show 10 balanced combinations for a model")
        print("4. Show baseline score for all models")
        print("5. Show all 11 single feature results for a model")
        print("6. Show all 11 single feature tuning results for a model")
        print("7. Show baseline tuned score for all models")
        print("0. Exit")
        choice = input("Choose an option: ").strip()
        if choice == "0":
            break
        elif choice == "1":
            show_best_combinations_all()
        elif choice in ["2", "3", "5", "6"]:
            model_key, model_label, model_index = select_model()
            if not model_key:
                print("❌ Invalid model number.")
                continue
            if choice == "2":
                show_top_10_combinations(model_key, model_label, model_index)
            elif choice == "3":
                show_balanced_combinations(model_key, model_label, model_index)
            elif choice == "5":
                show_single_feature_results(model_key, model_label, model_index)
            if choice == "6":
                show_single_tuned_results(model_key, model_label, model_index)
        elif choice == "4":
            show_baseline_scores()
        elif choice == "7":
            show_baseline_tuned_scores()
        else:
            print("❌ Invalid option.")


if __name__ == "__main__":
    stats_menu()
