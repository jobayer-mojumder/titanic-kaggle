# type: ignore
import os
import pandas as pd
from modules.feature_implementation import FEATURE_MAP
from modules.constant import MODEL_ORDER


def run_balanced_combinations(tune: bool = False):
    print(f"🔍 Mode: {'TUNING' if tune else 'NORMAL'}")

    base_dir = (
        "kaggle-results/tuning-combinations"
        if tune
        else "kaggle-results/features-combinations"
    )
    suffix = "_comb_tuned.csv" if tune else "_comb.csv"

    file_paths = {
        "Decision Tree": f"1_dt{suffix}",
        "XGBoost": f"2_xgb{suffix}",
        "Random Forest": f"3_rf{suffix}",
        "LightGBM": f"4_lgbm{suffix}",
        "CatBoost": f"5_cb{suffix}",
    }

    NAME_TO_NUM = {v: k for k, v in FEATURE_MAP.items()}
    balanced_samples = []

    for model, filename in file_paths.items():
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            print(f"⚠️ File not found: {path}")
            continue

        try:
            df = pd.read_csv(path)
            if (
                df.empty
                or "accuracy_vs_kaggle" not in df.columns
                or "features" not in df.columns
            ):
                print(f"⚠️ Invalid or empty data in: {path}")
                continue

            df = df.sort_values("accuracy_vs_kaggle").reset_index(drop=True)
            total = len(df)

            top = df.tail(3)
            middle = df.iloc[(total - 4) // 2 : (total - 4) // 2 + 4]
            bottom = df.head(3)

            sample = pd.concat([top, middle, bottom], ignore_index=True)
            sample["model_name"] = model

            def parse_features(feature_str):
                return sorted(
                    [
                        NAME_TO_NUM[f.strip()]
                        for f in feature_str.split(",")
                        if f.strip() in NAME_TO_NUM
                    ]
                )

            sample["feature_nums"] = (
                sample["features"].astype(str).apply(parse_features)
            )
            balanced_samples.append(sample)

        except Exception as e:
            print(f"❌ Error processing {path}: {e}")

    if balanced_samples:
        final_df = pd.concat(balanced_samples, ignore_index=True)

        final_df["model_order"] = final_df["model_name"].map(MODEL_ORDER)
        final_df["feature_len"] = final_df["feature_nums"].apply(len)
        final_df["feature_str"] = final_df["feature_nums"].apply(
            lambda x: ",".join(map(str, x))
        )

        final_df = final_df.sort_values(
            by=["model_order", "feature_len", "feature_str"]
        ).reset_index(drop=True)

        final_df.drop(
            columns=["model_order", "feature_len", "feature_str"], inplace=True
        )

        cols = [
            "model",
            "model_name",
            "features",
            "feature_nums",
            "accuracy_vs_kaggle",
            "improvement",
        ]
        final_df = final_df[
            [c for c in cols if c in final_df.columns]
            + [c for c in final_df.columns if c not in cols]
        ]

        print("\n✅ Final Balanced Sample:")
        print(final_df)

        os.makedirs("results", exist_ok=True)
        out_path = "results/balanced_combinations.csv"
        final_df.to_csv(out_path, index=False)
        print(f"\n💾 Saved to {out_path}")
    else:
        print("\n⚠️ No valid model files found. Nothing to save.")
