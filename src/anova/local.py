# anova/local.py

import pandas as pd
import ast
import os


def generate_jasp_ready_data(input_path, output_folder, expand_variance=False):
    # Load input CSV
    summary = pd.read_csv(input_path)

    # Prepare columns
    summary["feature_num"] = summary["feature_num"].astype(str)
    summary["tuned"] = summary["tuned"].fillna(0).astype(int)
    summary["model"] = summary["model"].astype(str)
    summary["accuracy"] = summary["accuracy"].fillna(0)
    summary["cv_scores"] = summary["cv_scores"].fillna("[]")

    expanded_rows = []

    for idx, row in summary.iterrows():
        model = row["model"]
        feature_num = row["feature_num"]
        tuned = row["tuned"]
        acc_mean = row["accuracy"]
        cv_scores_str = row.get("cv_scores", "[]")

        # Safe parse cv_scores
        try:
            fold_scores = ast.literal_eval(cv_scores_str)
            if not isinstance(fold_scores, list):
                fold_scores = []
        except (ValueError, SyntaxError):
            fold_scores = []

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

        if feature_num == "baseline":
            if fold_scores:
                for fold_score in fold_scores:
                    expanded_rows.append(
                        {"accuracy": fold_score, "model": model, "group": group}
                    )
            else:
                expanded_rows.append(
                    {"accuracy": acc_mean, "model": model, "group": group}
                )
        else:
            if expand_variance and fold_scores:
                for fold_score in fold_scores:
                    expanded_rows.append(
                        {"accuracy": fold_score, "model": model, "group": group}
                    )
            else:
                expanded_rows.append(
                    {"accuracy": acc_mean, "model": model, "group": group}
                )

    # Create DataFrame
    expanded_df = pd.DataFrame(expanded_rows)

    # Force categorical types (optional, but recommended)
    expanded_df["model"] = expanded_df["model"].astype("category")
    expanded_df["group"] = expanded_df["group"].astype("category")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save
    filename = "jasp_local_variance.csv" if expand_variance else "jasp_local.csv"
    output_path = os.path.join(output_folder, filename)
    expanded_df.to_csv(output_path, index=False)

    print(f"✅ Saved clean file to {output_path}")
