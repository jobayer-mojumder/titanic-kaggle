# anova/local.py

import pandas as pd
import ast
import os


def generate_jasp_ready_data(input_path, output_folder, expand_variance=False):
    summary = pd.read_csv(input_path)

    # Standardize and clean columns
    summary["feature_num"] = summary["feature_num"].astype(str)
    summary["tuned"] = summary["tuned"].fillna(0).astype(int)
    summary["model"] = summary["model"].astype(str)
    summary["accuracy"] = summary["accuracy"].fillna(0)
    summary["cv_scores"] = summary["cv_scores"].fillna("[]")

    expanded_rows = []

    for _, row in summary.iterrows():
        model = row["model"]
        feature_num = row["feature_num"]
        tuned = int(row["tuned"])
        acc_mean = row["accuracy"]
        cv_scores_str = row["cv_scores"]

        # Determine group labels
        feature_eng = 0 if feature_num == "baseline" else 1
        model_tuning = tuned

        # Safe parse CV scores
        try:
            fold_scores = ast.literal_eval(cv_scores_str)
            if not isinstance(fold_scores, list):
                fold_scores = []
        except Exception:
            fold_scores = []

        if feature_eng == 0:
            if fold_scores:
                for fold_score in fold_scores:
                    expanded_rows.append(
                        {
                            "accuracy": fold_score,
                            "Feature_Engineering": feature_eng,
                            "Model_Tuning": model_tuning,
                            "model": model,
                        }
                    )
            else:
                expanded_rows.append(
                    {
                        "accuracy": acc_mean,
                        "Feature_Engineering": feature_eng,
                        "Model_Tuning": model_tuning,
                        "model": model,
                    }
                )

        else:
            if expand_variance and fold_scores:
                for score in fold_scores:
                    expanded_rows.append(
                        {
                            "model": model,
                            "Feature_Engineering": feature_eng,
                            "Model_Tuning": model_tuning,
                            "accuracy": score,
                        }
                    )
            else:
                expanded_rows.append(
                    {
                        "model": model,
                        "Feature_Engineering": feature_eng,
                        "Model_Tuning": model_tuning,
                        "accuracy": acc_mean,
                    }
                )

    # Convert to DataFrame
    df_out = pd.DataFrame(expanded_rows)

    # Save
    os.makedirs(output_folder, exist_ok=True)
    filename = "anova_local_variance.csv" if expand_variance else "anova_local.csv"
    output_path = os.path.join(output_folder, filename)
    df_out.to_csv(output_path, index=False)

    print(f"✅ Saved ANOVA-ready data to {output_path}")
