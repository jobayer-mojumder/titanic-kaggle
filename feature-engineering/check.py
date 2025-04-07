# type: ignore
import pandas as pd
from itertools import combinations

for i in range(1, 12):
    models = {
        "dt": pd.read_csv(f"submissions/1_dt/submission_dt_{i}.csv")["Survived"],
        "xgb": pd.read_csv(f"submissions/2_xgb/submission_xgb_{i}.csv")["Survived"],
        "rf": pd.read_csv(f"submissions/3_rf/submission_rf_{i}.csv")["Survived"],
        "lgbm": pd.read_csv(f"submissions/4_lgbm/submission_lgbm_{i}.csv")["Survived"],
        "cb": pd.read_csv(f"submissions/5_cb/submission_cb_{i}.csv")["Survived"],
    }

    # Group models with identical predictions
    matched_groups = []
    visited = set()

    for name, preds in models.items():
        if name in visited:
            continue
        group = [name]
        visited.add(name)
        for other_name, other_preds in models.items():
            if other_name not in visited and (preds == other_preds).all():
                group.append(other_name)
                visited.add(other_name)
        if len(group) > 1:
            matched_groups.append(group)

    if matched_groups:
        print(f"\n🔍 Feature #{i} — Matching Predictions:")
        for group in matched_groups:
            print("✅ " + " == ".join(group))
