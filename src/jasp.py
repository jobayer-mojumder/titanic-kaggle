import pandas as pd
import os
import ast

# Load local summary
local_df = pd.read_csv("results/summary_local.csv")


def save_csv(df, filename):
    out_dir = "jasp_outputs/local"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"✅ Saved: {path}")


def extract_cv_scores(row):
    if isinstance(row, str):
        try:
            return ast.literal_eval(row)
        except Exception:
            return []
    return []


def prepare_and_save_per_model_comparisons(local_df):
    for model in local_df["model"].unique():
        df_model = local_df[local_df["model"] == model].copy()

        # ---- Extract baseline CV scores ----
        baseline_row = df_model[
            (df_model["feature_num"] == "baseline") & (df_model["tuned"] == 0)
        ]
        baseline_scores = []
        for _, row in baseline_row.iterrows():
            baseline_scores += extract_cv_scores(row.get("cv_scores", ""))

        baseline_df = pd.DataFrame(
            {"model": model, "group": "baseline", "score": baseline_scores}
        )

        # ---- Feature Engineering Only ----
        fe_only_df = df_model[
            (df_model["feature_num"] != "baseline") & (df_model["tuned"] == 0)
        ]
        fe_rows = pd.DataFrame(
            {"model": model, "group": "feature_eng", "score": fe_only_df["accuracy"]}
        )

        fe_combined = pd.concat([baseline_df, fe_rows], ignore_index=True)
        if fe_combined["group"].value_counts().min() >= 2:
            save_csv(
                fe_combined[["model", "group", "score"]],
                f"local_{model.lower()}_fe_vs_baseline.csv",
            )

        # ---- Model Tuning Only ----
        tuned_baseline_row = df_model[
            (df_model["feature_num"] == "baseline") & (df_model["tuned"] == 1)
        ]
        tuning_scores = []
        for _, row in tuned_baseline_row.iterrows():
            tuning_scores += extract_cv_scores(row.get("cv_scores", ""))

        tuning_df = pd.DataFrame(
            {"model": model, "group": "tuning", "score": tuning_scores}
        )

        tuning_combined = pd.concat([baseline_df, tuning_df], ignore_index=True)
        if tuning_combined["group"].value_counts().min() >= 2:
            save_csv(
                tuning_combined[["model", "group", "score"]],
                f"local_{model.lower()}_tuning_vs_baseline.csv",
            )

        # ---- Feature Engineering + Tuning ----
        fe_mt_df = df_model[
            (df_model["feature_num"] != "baseline") & (df_model["tuned"] == 1)
        ]
        fe_mt_rows = pd.DataFrame(
            {"model": model, "group": "fe_tuning", "score": fe_mt_df["accuracy"]}
        )

        fe_mt_combined = pd.concat([baseline_df, fe_mt_rows], ignore_index=True)
        if fe_mt_combined["group"].value_counts().min() >= 2:
            save_csv(
                fe_mt_combined[["model", "group", "score"]],
                f"local_{model.lower()}_fe_mt_vs_baseline.csv",
            )


# 🚀 Run it!
prepare_and_save_per_model_comparisons(local_df)
