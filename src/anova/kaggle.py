import pandas as pd
import os


def generate_jasp_ready_data_kaggle(input_path, output_folder):
    summary = pd.read_csv(input_path)

    # Standardize and clean columns
    summary["feature_num"] = summary["feature_num"].astype(str)
    summary["tuned"] = summary["tuned"].fillna(0).astype(int)
    summary["model"] = summary["model"].astype(str)
    summary["kaggle_score"] = summary["kaggle_score"].fillna(0)

    expanded_rows = []

    for _, row in summary.iterrows():
        model = row["model"]
        feature_num = row["feature_num"]
        tuned = int(row["tuned"])
        kaggle_score = row["kaggle_score"]

        # Determine binary flags
        feature_eng = 0 if feature_num == "baseline" else 1
        model_tuning = tuned

        # No variance, so only one row per experiment
        expanded_rows.append(
            {
                "model": model,
                "Feature_Engineering": feature_eng,
                "Model_Tuning": model_tuning,
                "accuracy": kaggle_score,
            }
        )

    # Convert to DataFrame
    df_out = pd.DataFrame(expanded_rows)

    # Save
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "anova_kaggle.csv")
    df_out.to_csv(output_path, index=False)

    print(f"✅ Saved ANOVA-ready Kaggle data to {output_path}")
