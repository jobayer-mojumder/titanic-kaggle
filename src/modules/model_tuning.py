from sklearn.model_selection import GridSearchCV, ParameterGrid
from modules.constant import DEFAULT_MODELS

PARAM_GRIDS = {
    "dt": {
        "criterion": ["gini", "entropy"],
        "splitter": ["best"],
        "max_depth": [2, 3, 4],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
    },
    "rf": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 4, 5],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
        "bootstrap": [True],
        "max_features": ["sqrt"],
    },
    "xgb": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.1],
        "min_child_weight": [3, 5],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_alpha": [0.1],
        "reg_lambda": [1],
    },
    "lgbm": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [15, 31],
        "min_child_samples": [20, 30],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_alpha": [0.1],
        "reg_lambda": [1],
    },
    "cb": {
        "iterations": [50, 100, 200],
        "depth": [3, 4],
        "learning_rate": [0.05, 0.1],
        "l2_leaf_reg": [3, 5],
        "bagging_temperature": [0],
        "random_strength": [1],
    },
}


def tune_model(X, y, model_key, cv=3, scoring="accuracy"):
    from sklearn.model_selection import GridSearchCV, ParameterGrid
    from modules.constant import DEFAULT_MODELS
    import pandas as pd
    import numpy as np
    import scipy.sparse as sp

    print(f"🔍 Tuning model: {model_key.upper()}")

    def ensure_numeric_features(X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            if isinstance(X[col].iloc[0], sp.spmatrix):
                expanded = pd.DataFrame(
                    np.vstack(X[col].apply(lambda x: x.toarray().ravel())),
                    index=X.index,
                )
                expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]
                X = pd.concat([X.drop(columns=[col]), expanded], axis=1)

        non_numeric = X.select_dtypes(include=["object", "category"]).columns
        if len(non_numeric) > 0:
            X = pd.get_dummies(X, columns=non_numeric, drop_first=True)

        return X

    model = DEFAULT_MODELS[model_key]
    raw_grid = PARAM_GRIDS[model_key]

    # Ensure all values are lists
    param_grid = {k: v if isinstance(v, list) else [v] for k, v in raw_grid.items()}
    all_combos = list(ParameterGrid(param_grid))

    # 🧠 Klingert filtering logic
    filtered = []
    for combo in all_combos:
        max_depth = combo.get("max_depth")
        min_leaf = combo.get("min_samples_leaf", 1)
        if isinstance(max_depth, int) and max_depth > 4 and min_leaf >= 4:
            continue
        filtered.append(combo)

    filtered_out = len(all_combos) - len(filtered)
    print(f"⚠️  Filtered out {filtered_out} invalid combinations.")

    if not filtered:
        raise ValueError("❌ All parameter combinations were filtered out.")

    # ✅ Safe rebuilding of param grid (handle NoneType + strings safely)
    filtered_param_grid = {}
    for k in param_grid:
        values = {combo[k] for combo in filtered if k in combo}
        if None in values:
            sorted_values = sorted([v for v in values if v is not None])
            filtered_param_grid[k] = sorted_values + [None]
        else:
            filtered_param_grid[k] = sorted(values)

    print(
        f"✅ Using {len(list(ParameterGrid(filtered_param_grid)))} tuning combinations."
    )

    X = ensure_numeric_features(X)
    grid = GridSearchCV(
        estimator=model,
        param_grid=filtered_param_grid,
        cv=cv,
        scoring=scoring,
        verbose=1,
        n_jobs=-1,
    )

    grid.fit(X, y)

    print(f"✅ Best params: {grid.best_params_}")
    print(f"📈 Best score: {grid.best_score_:.5f}")

    return grid.best_estimator_, grid.best_params_
