from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV
from modules.constant import DEFAULT_MODELS, PARAM_GRIDS
import pandas as pd
import numpy as np
import scipy.sparse as sp


def tune_model(X, y, model_key, cv=5, scoring="accuracy"):
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

    X = ensure_numeric_features(X)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=25,
        cv=cv,
        scoring="accuracy",
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )

    search.fit(X, y)

    print(f"📈 Best score: {search.best_score_:.5f}")

    return search.best_estimator_, search.best_params_
