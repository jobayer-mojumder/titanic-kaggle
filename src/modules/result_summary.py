# type: ignore
import pandas as pd
import os
from modules.feature_implementation import FEATURE_MAP
from modules.constant import MODEL_ORDER

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


def run_best_results(tune: bool = False):
    print(f"{CYAN}🔍 Mode: {'TUNING' if tune else 'NORMAL'}{RESET}")

    file_paths = {
        "Decision Tree": "1_dt_comb.csv",
        "XGBoost": "2_xgb_comb.csv",
        "Random Forest": "3_rf_comb.csv",
        "LightGBM": "4_lgbm_comb.csv",
        "CatBoost": "5_cb_comb.csv",
    }

    NAME_TO_NUM = {v: k for k, v in FEATURE_MAP.items()}
    base_dir = (
        "kaggle-results/tuning-combinations"
        if tune
        else "kaggle-results/features-combinations"
    )
    output_file = "results/best_results.csv"
    best_results = []

    for model, filename in file_paths.items():
        file_name = filename.replace("_comb", "_comb_tuned") if tune else filename
        file_path = os.path.join(base_dir, file_name)

        if not os.path.exists(file_path):
            print(f"{YELLOW}⚠️  File not found: {file_path}{RESET}")
            continue

        try:
            df = pd.read_csv(file_path)

            if df.empty:
                print(f"{YELLOW}⚠️  File is empty: {file_path}{RESET}")
                continue

            if "accuracy_vs_kaggle" not in df.columns:
                print(
                    f"{RED}❌ Missing column 'accuracy_vs_kaggle' in: {file_path}{RESET}"
                )
                continue

            best_row = df.loc[df["accuracy_vs_kaggle"].idxmax()].copy()
            best_row["model_name"] = model
            best_row["model"] = filename.split("_")[0]
            best_row["model_order"] = MODEL_ORDER.get(model, 999)

            def parse_features(feature_str):
                return sorted(
                    [
                        NAME_TO_NUM[f.strip()]
                        for f in feature_str.split(",")
                        if f.strip() in NAME_TO_NUM
                    ]
                )

            best_row["feature_nums"] = parse_features(str(best_row["features"]))
            best_results.append(best_row)

        except pd.errors.EmptyDataError:
            print(f"{YELLOW}⚠️  No data in file: {file_path}{RESET}")
        except Exception as e:
            print(f"{RED}❌ Error processing {file_path}: {e}{RESET}")

    if best_results:
        best_df = pd.DataFrame(best_results)
        best_df = best_df.sort_values(by=["model_order"]).reset_index(drop=True)

        preferred_cols = [
            "model",
            "model_name",
            "features",
            "feature_nums",
            "accuracy_vs_kaggle",
            "improvement",
        ]
        other_cols = [
            c for c in best_df.columns if c not in preferred_cols + ["model_order"]
        ]
        best_df = best_df[preferred_cols + other_cols]

        print(f"\n{GREEN}✅ Best results per model:{RESET}")
        print(best_df.to_string(index=False))

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        best_df.to_csv(output_file, index=False)
        print(f"\n{BLUE}💾 Saved to {output_file}{RESET}")
    else:
        print(f"\n{YELLOW}⚠️ No valid results found to save.{RESET}")
