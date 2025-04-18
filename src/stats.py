import os
import pandas as pd
from tabulate import tabulate
from modules.analysis import (
    get_best_single_feature_combination,
    get_10_best_feature_combinations,
    get_10_balanced_feature_combinations,
)

MODEL_KEYS = {
    "1": ("dt", "Decision Tree", 1),
    "2": ("xgb", "XGBoost", 2),
    "3": ("rf", "Random Forest", 3),
    "4": ("lgbm", "LightGBM", 4),
    "5": ("cb", "CatBoost", 5),
}
MODEL_KEYS_REV = {v[0]: k for k, v in MODEL_KEYS.items()}


def choose_mode():
    os.system("cls" if os.name == "nt" else "clear")
    print("\n📁 Choose result mode:")
    print("  [1] Kaggle (default)")
    print("  [2] Local")
    mode = input("Select mode (default Kaggle): ").strip()
    return "local" if mode == "2" else "kaggle"


def select_model():
    print("\n🔧 Choose a model:")
    for k, (_, name, _) in MODEL_KEYS.items():
        print(f"{k}. {name}")
    choice = input("Enter model number: ").strip()
    return MODEL_KEYS.get(choice, (None, None, None))


def display_table(df, title, file_name, mode):
    os.system("cls" if os.name == "nt" else "clear")
    if df.empty:
        print("⚠️ No data found.")
        return
    print(f"\n📊 {title}")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))

    out_dir = os.path.join("stats", mode)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, file_name)
    df.to_csv(output_path, index=False)
    print(f"\n📁 Data saved to {output_path}")
    input("Press Enter to continue...")


def extract_rows_for_combos(model_key, combos, tuned=False, mode="kaggle"):
    folder = "tuning-combinations" if tuned else "features-combinations"
    suffix = "_comb_tuned.csv" if tuned else "_comb.csv"
    index = MODEL_KEYS_REV[model_key]
    path = f"results/{mode}/{folder}/{index}_{model_key}{suffix}"
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    df["feature_nums"] = df["feature_nums"].astype(str).str.strip()
    if tuned and "params" in df.columns:
        df["params"] = df["params"].fillna("").astype(str)
    rows = []
    for combo in combos:
        combo_str = ", ".join(map(str, sorted(combo)))
        match = df[df["feature_nums"] == combo_str]
        if not match.empty:
            rows.append(match.iloc[0].to_dict())
    return rows


def extract_single_feature_scores(model_key, model_index, tuned=False, mode="kaggle"):
    folder = "single-tuning" if tuned else "single-features"
    suffix = "single_tuned.csv" if tuned else "single.csv"
    filename = f"{model_index}_{model_key}_{suffix}"
    path = os.path.join("results", mode, folder, filename)
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print("⚠️ No data found.")
        return None
    df["rank"] = df[score_column(mode)].rank(method="min", ascending=False).astype(int)
    df = df.sort_values(by=score_column(mode), ascending=False)
    if tuned and "params" in df.columns:
        df["params"] = df["params"].fillna("").astype(str)
    return df


def extract_baseline_scores(tuned=False, mode="kaggle"):
    summary_path = f"results/summary_{mode}.csv"
    if not os.path.exists(summary_path):
        print(f"❌ Summary file not found at {summary_path}")
        return None
    df = pd.read_csv(summary_path)
    filtered = df[
        (df["feature_nums"] == "baseline") & (df["tuned"] == (1 if tuned else 0))
    ]
    rows = []
    for _, (model_key, model_label, _) in MODEL_KEYS.items():
        row = filtered[filtered["model"] == model_key]
        if not row.empty:
            score = row.iloc[0][score_column(mode)]
            row_data = [model_label, model_key, round(score, 5)]
            if tuned and "params" in row.columns:
                row_data.append(row.iloc[0].get("params", ""))
            rows.append(row_data)
        else:
            rows.append([model_label, model_key, "Not found"])
    columns = [
        "Model",
        "Key",
        f"{'Tuned' if tuned else 'Untuned'} Baseline {mode.title()} Score",
    ]
    if tuned and "params" in df.columns:
        columns.append("params")
    return pd.DataFrame(rows, columns=columns)


def score_column(mode):
    return "kaggle_score" if mode == "kaggle" else "accuracy"


def show_combinations(
    model_key,
    model_label,
    model_index,
    combos,
    title,
    filename,
    tuned=False,
    mode="kaggle",
):
    rows = extract_rows_for_combos(model_key, combos, tuned=tuned, mode=mode)
    if not rows:
        print("⚠️ No matching rows found.")
        return
    df = pd.DataFrame(rows)
    df.insert(0, "rank", range(1, len(df) + 1))
    cols = ["rank", "feature_nums", score_column(mode), "improvement", "tuned"]
    if tuned and "params" in df.columns:
        cols.append("params")
    df = df[cols]
    display_table(df, title, filename, mode)


def print_menu(mode):
    os.system("cls" if os.name == "nt" else "clear")
    print("\n" + "=" * 50)
    print(f"🎯 Stats Menu - Mode: [{mode.upper()}]".center(50))
    print("=" * 50)
    print("\n📊 Feature Engineering")
    print("     [1]  Best feature combination for all models")
    print("     [2]  Top 10 feature combinations for a model")
    print("     [3]  Balanced 10 feature combinations for a model")
    print("     [4]  Single feature results for a model")
    print("     [5]  Baseline score (untuned) for all models")
    print("\n🔧 Model Tuning")
    print("     [6]  Tuned single feature results for a model")
    print("     [7]  Tuned baseline score for all models")
    print("     [8]  Tuned best combination (from FE)")
    print("     [9]  Tuned top 10 combinations (from FE)")
    print("     [10] Tuned balanced 10 combinations (from FE)")
    print("\n📊")
    print("     [11] Separate single feature summary for all models (Untuned)")
    print("     [12] Separate single feature summary for all models (Tuned)")
    print("\n🧪 Utility")
    print("     [99] Run all reports for all models")
    print("\n⚙️ Settings")
    print("    [m]  Change mode")
    print("    [0]  Exit")
    print("=" * 50)


def stats_menu():
    global mode
    mode = "kaggle"
    while True:
        print_menu(mode)
        choice = input("Choose an option: ").strip()
        if choice == "0":
            break
        elif choice.lower() == "m":
            mode = choose_mode()
            continue
        elif choice == "1":
            from modules.analysis import get_best_single_feature_combination

            rows = []
            for _, (model_key, model_label, model_index) in MODEL_KEYS.items():
                combo = get_best_single_feature_combination(model_key, mode=mode)
                result = extract_rows_for_combos(
                    model_key, [combo], tuned=False, mode=mode
                )
                if result:
                    rows.append(result[0])
            if rows:
                df = pd.DataFrame(rows)
                df.insert(0, "rank", range(1, len(df) + 1))
                df = df[
                    ["rank", "feature_nums", score_column(mode), "improvement", "tuned"]
                ]
                display_table(
                    df,
                    "Best Feature Combination for All Models",
                    "best_combinations.csv",
                    mode,
                )
        elif choice == "2":
            model_key, model_label, model_index = select_model()
            if model_key:
                combos = get_10_best_feature_combinations(model_key, mode=mode)
                show_combinations(
                    model_key,
                    model_label,
                    model_index,
                    combos,
                    f"Top 10 Feature Combinations for {model_label}",
                    f"{model_index}_{model_key}_top10.csv",
                    tuned=False,
                    mode=mode,
                )
        elif choice == "3":
            model_key, model_label, model_index = select_model()
            if model_key:
                combos = get_10_balanced_feature_combinations(model_key, mode=mode)
                show_combinations(
                    model_key,
                    model_label,
                    model_index,
                    combos,
                    f"Balanced 10 Feature Combinations for {model_label}",
                    f"{model_index}_{model_key}_balanced10.csv",
                    tuned=False,
                    mode=mode,
                )
        elif choice == "4":
            model_key, model_label, model_index = select_model()
            if model_key:
                df = extract_single_feature_scores(
                    model_key, model_index, tuned=False, mode=mode
                )
                if df is not None:
                    cols = [
                        "rank",
                        "feature_nums",
                        score_column(mode),
                        "improvement",
                        "tuned",
                    ]
                    df = df[cols]
                    display_table(
                        df,
                        f"Single Feature Results for {model_label}",
                        f"{model_index}_{model_key}_single_features.csv",
                        mode,
                    )
        elif choice == "5":
            df = extract_baseline_scores(tuned=False, mode=mode)
            if df is not None:
                display_table(
                    df,
                    "Baseline Scores (Untuned) for All Models",
                    f"baseline_scores_{mode}_untuned.csv",
                    mode,
                )
        elif choice == "6":
            model_key, model_label, model_index = select_model()
            if model_key:
                df = extract_single_feature_scores(
                    model_key, model_index, tuned=True, mode=mode
                )
                if df is not None:
                    cols = [
                        "rank",
                        "feature_nums",
                        score_column(mode),
                        "improvement",
                        "tuned",
                    ]
                    if "params" in df.columns:
                        cols.append("params")
                    df = df[cols]
                    display_table(
                        df,
                        f"Tuned Single Feature Results for {model_label}",
                        f"{model_index}_{model_key}_single_tuned.csv",
                        mode,
                    )
        elif choice == "7":
            df = extract_baseline_scores(tuned=True, mode=mode)
            if df is not None:
                display_table(
                    df,
                    "Baseline Scores (Tuned) for All Models",
                    f"baseline_scores_{mode}_tuned.csv",
                    mode,
                )
        elif choice in ["8", "9", "10"]:
            model_key, model_label, model_index = select_model()
            if model_key:
                if choice == "8":
                    combos = [get_best_single_feature_combination(model_key, mode=mode)]
                    title = f"Tuned Best Combination for {model_label}"
                    fname = f"{model_index}_{model_key}_best_tuned.csv"
                elif choice == "9":
                    combos = get_10_best_feature_combinations(model_key, mode=mode)
                    title = f"Tuned Top 10 Combinations for {model_label}"
                    fname = f"{model_index}_{model_key}_top10_tuned.csv"
                else:
                    combos = get_10_balanced_feature_combinations(model_key, mode=mode)
                    title = f"Tuned Balanced 10 Combinations for {model_label}"
                    fname = f"{model_index}_{model_key}_balanced10_tuned.csv"
                show_combinations(
                    model_key,
                    model_label,
                    model_index,
                    combos,
                    title,
                    fname,
                    tuned=True,
                    mode=mode,
                )
        elif choice in ["11", "12"]:
            tuned = choice == "12"
            out_dir = os.path.join("stats", mode, "single")
            os.makedirs(out_dir, exist_ok=True)
            for feature_num in range(1, 12):
                rows = []
                for k, (model_key, model_label, model_index) in MODEL_KEYS.items():
                    df = extract_single_feature_scores(
                        model_key, model_index, tuned=tuned, mode=mode
                    )
                    if df is not None:
                        df["feature_nums"] = df["feature_nums"].astype(str).str.strip()
                        match = df[df["feature_nums"] == str(feature_num)]
                        if not match.empty:
                            row = match.iloc[0]
                            row_data = {
                                "model": model_label,
                                "model_key": model_key,
                                "feature_num": feature_num,
                                score_column(mode): row[score_column(mode)],
                                "improvement": row.get("improvement", ""),
                                "tuned": 1 if tuned else 0,
                            }
                            if "params" in row:
                                row_data["params"] = row["params"]
                            rows.append(row_data)
                if rows:
                    out_df = pd.DataFrame(rows)
                    filename = f"single_feature_{feature_num}_{mode}{'_tuned' if tuned else ''}.csv"
                    out_path = os.path.join(out_dir, filename)
                    out_df.to_csv(out_path, index=False)
                    display_table(
                        out_df,
                        f"Feature {feature_num} Results Across Models ({'Tuned' if tuned else 'Untuned'})",
                        filename,
                        os.path.join(mode, "single"),
                    )
        elif choice == "99":
            print("🧪 Running all 12 reports for all models...\n")
            for auto_choice in [str(i) for i in range(1, 13)]:
                print(f"\n▶️ Running option [{auto_choice}]")
                run_menu_choice(auto_choice, mode)
            print("\n✅ All reports completed.")
            input("Press Enter to return to menu...")


def run_menu_choice(choice, mode):
    if choice == "1":
        for _, (model_key, model_label, model_index) in MODEL_KEYS.items():
            combo = get_best_single_feature_combination(model_key, mode=mode)
            rows = extract_rows_for_combos(model_key, [combo], tuned=False, mode=mode)
            if rows:
                df = pd.DataFrame(rows)
                df.insert(0, "rank", 1)
                df = df[
                    ["rank", "feature_nums", score_column(mode), "improvement", "tuned"]
                ]
                display_table(
                    df,
                    f"Best Combination for {model_label}",
                    f"{model_index}_{model_key}_best.csv",
                    mode,
                )

    elif choice in ["2", "3", "4", "6", "9", "10"]:
        tuned = choice in ["6", "9", "10"]
        for _, (model_key, model_label, model_index) in MODEL_KEYS.items():
            if choice in ["2", "9"]:
                combos = get_10_best_feature_combinations(model_key, mode=mode)
            elif choice in ["3", "10"]:
                combos = get_10_balanced_feature_combinations(model_key, mode=mode)

            if choice in ["2", "3", "9", "10"]:
                title = f"{'Tuned ' if tuned else ''}{'Top' if choice in ['2','9'] else 'Balanced'} 10 for {model_label}"
                fname = f"{model_index}_{model_key}_{'top10' if choice in ['2','9'] else 'balanced10'}{'_tuned' if tuned else ''}.csv"
                show_combinations(
                    model_key,
                    model_label,
                    model_index,
                    combos,
                    title,
                    fname,
                    tuned=tuned,
                    mode=mode,
                )

            elif choice in ["4", "6"]:
                df = extract_single_feature_scores(
                    model_key, model_index, tuned=tuned, mode=mode
                )
                if df is not None:
                    cols = [
                        "rank",
                        "feature_nums",
                        score_column(mode),
                        "improvement",
                        "tuned",
                    ]
                    if tuned and "params" in df.columns:
                        cols.append("params")
                    df = df[cols]
                    filename = f"{model_index}_{model_key}_{'single_tuned' if tuned else 'single_features'}.csv"
                    display_table(
                        df,
                        f"{'Tuned ' if tuned else ''}Single Feature Results for {model_label}",
                        filename,
                        mode,
                    )

    elif choice in ["5", "7"]:
        tuned = choice == "7"
        df = extract_baseline_scores(tuned=tuned, mode=mode)
        if df is not None:
            filename = f"baseline_scores_{mode}{'_tuned' if tuned else '_untuned'}.csv"
            display_table(
                df,
                f"{'Tuned ' if tuned else 'Untuned'} Baseline Scores",
                filename,
                mode,
            )

    elif choice in ["8"]:
        for _, (model_key, model_label, model_index) in MODEL_KEYS.items():
            combo = get_best_single_feature_combination(model_key, mode=mode)
            show_combinations(
                model_key,
                model_label,
                model_index,
                [combo],
                f"Tuned Best Combination for {model_label}",
                f"{model_index}_{model_key}_best_tuned.csv",
                tuned=True,
                mode=mode,
            )

    elif choice in ["11", "12"]:
        tuned = choice == "12"
        out_dir = os.path.join("stats", mode, "single")
        os.makedirs(out_dir, exist_ok=True)
        for feature_num in range(1, 12):
            rows = []
            for _, (model_key, model_label, model_index) in MODEL_KEYS.items():
                df = extract_single_feature_scores(
                    model_key, model_index, tuned=tuned, mode=mode
                )
                if df is not None:
                    df["feature_nums"] = df["feature_nums"].astype(str).str.strip()
                    match = df[df["feature_nums"] == str(feature_num)]
                    if not match.empty:
                        row = match.iloc[0]
                        row_data = {
                            "model": model_label,
                            "model_key": model_key,
                            "feature_num": feature_num,
                            score_column(mode): row[score_column(mode)],
                            "improvement": row.get("improvement", ""),
                            "tuned": 1 if tuned else 0,
                        }
                        if "params" in row:
                            row_data["params"] = row["params"]
                        rows.append(row_data)
            if rows:
                out_df = pd.DataFrame(rows)
                filename = f"single_feature_{feature_num}_{mode}{'_tuned' if tuned else ''}.csv"
                out_path = os.path.join(out_dir, filename)
                out_df.to_csv(out_path, index=False)
                display_table(
                    out_df,
                    f"Feature {feature_num} Across Models ({'Tuned' if tuned else 'Untuned'})",
                    filename,
                    os.path.join(mode, "single"),
                )


if __name__ == "__main__":
    stats_menu()
