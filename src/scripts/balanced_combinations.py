# type: ignore
import pandas as pd
import os

# File paths for each model
file_paths = {
    "Decision Tree": "1_dt_comb.csv",
    "XGBoost": "2_xgb_comb.csv",
    "Random Forest": "3_rf_comb.csv",
    "LightGBM": "4_lgbm_comb.csv",
    "CatBoost": "5_cb_comb.csv",
}

# Feature name to number map
FEATURE_MAP = {
    1: "title",
    2: "family_size",
    3: "is_alone",
    4: "age_group",
    5: "fare_per_person",
    6: "deck",
    7: "has_cabin",
    8: "is_mother",
    9: "sex_pclass",
    10: "is_child",
    11: "women_children_first",
}
NAME_TO_NUM = {v: k for k, v in FEATURE_MAP.items()}

# Model sort order
model_order = {
    "Decision Tree": 0,
    "XGBoost": 1,
    "Random Forest": 2,
    "LightGBM": 3,
    "CatBoost": 4,
}

# Store processed rows
balanced_samples = []

for model, filename in file_paths.items():
    path = os.path.join("../combination-results", filename)

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

        # Select top 3, middle 4, bottom 3
        top = df.tail(3)
        middle = df.iloc[(total - 4) // 2 : (total - 4) // 2 + 4]
        bottom = df.head(3)

        # Combine and annotate
        sample = pd.concat([top, middle, bottom], ignore_index=True)
        sample["model_name"] = model

        # Convert features to numeric IDs
        def parse_features(feature_str):
            return sorted(
                [
                    NAME_TO_NUM[f.strip()]
                    for f in feature_str.split(",")
                    if f.strip() in NAME_TO_NUM
                ]
            )

        sample["feature_nums"] = sample["features"].astype(str).apply(parse_features)
        balanced_samples.append(sample)

    except Exception as e:
        print(f"❌ Error processing {path}: {e}")

# Final combination
if balanced_samples:
    final_df = pd.concat(balanced_samples, ignore_index=True)

    # Add helper sort columns
    final_df["model_order"] = final_df["model_name"].map(model_order)
    final_df["feature_len"] = final_df["feature_nums"].apply(len)
    final_df["feature_str"] = final_df["feature_nums"].apply(
        lambda x: ",".join(map(str, x))
    )

    # Sort
    final_df = final_df.sort_values(
        by=["model_order", "feature_len", "feature_str"]
    ).reset_index(drop=True)

    # Drop helpers
    final_df = final_df.drop(columns=["model_order", "feature_len", "feature_str"])

    # re-arrange columns
    cols = [
        "model",
        "model_name",
        "features",
        "feature_nums",
        "accuracy_vs_kaggle",
        "improvement",
    ]
    cols += [col for col in final_df.columns if col not in cols]
    final_df = final_df[cols]

    # Show and save
    print("\n✅ Final Balanced Sample:")
    print(final_df)

    os.makedirs("../results", exist_ok=True)
    final_df.to_csv("../results/balanced_combinations.csv", index=False)
    print("\n💾 Saved to ../results/balanced_combinations.csv")
else:
    print("\n⚠️ No valid model files found. Nothing to save.")
