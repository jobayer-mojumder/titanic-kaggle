# anova/kaggle.py

import pandas as pd
import os


def generate_jasp_ready_data_kaggle(input_path, output_folder):
    # Load Kaggle summary CSV
    summary = pd.read_csv(input_path)

    # Prepare columns
    summary["feature_num"] = summary["feature_num"].astype(str)
    summary["tuned"] = summary["tuned"].fillna(0).astype(int)
    summary["model"] = summary["model"].astype(str)
    summary["kaggle_score"] = summary["kaggle_score"].fillna(0)

    expanded_rows = []

    for idx, row in summary.iterrows():
        model = row["model"]
        feature_num = row["feature_num"]
        tuned = row["tuned"]
        kaggle_score = row["kaggle_score"]

        # Determine group
        if feature_num == "baseline" and tuned == 0:
            group = "Baseline"
        elif feature_num != "baseline" and tuned == 0:
            group = "FE"
        elif feature_num == "baseline" and tuned == 1:
            group = "MT"
        elif feature_num != "baseline" and tuned == 1:
            group = "FE+MT"
        else:
            group = "Other"

        expanded_rows.append(
            {
                "accuracy": kaggle_score,
                "model": model,
                "group": group,
            }
        )

    # Create DataFrame
    expanded_df = pd.DataFrame(expanded_rows)

    # Force categorical types (optional)
    expanded_df["model"] = expanded_df["model"].astype("category")
    expanded_df["group"] = expanded_df["group"].astype("category")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save
    output_path = os.path.join(output_folder, "jasp_kaggle.csv")
    expanded_df.to_csv(output_path, index=False)

    print(f"✅ Saved Kaggle JASP-ready data to {output_path}")
