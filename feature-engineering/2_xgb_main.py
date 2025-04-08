# type: ignore
import os
import pandas as pd
import xgboost as xgb
from modules.preprocessing import preprocess
from modules.feature_implementation import FEATURE_MAP
from modules.summary import log_results
from modules.evaluation import evaluate_model


def run_model(feature_nums, use_cv=True):
    selected_features = [FEATURE_MAP[n] for n in feature_nums]
    print(f"🚀 Running xgb with: {selected_features or 'Baseline only'}")

    train = pd.read_csv("../train.csv")
    test = pd.read_csv("../test.csv")

    y = train["Survived"]
    train.drop(columns=["Survived"], inplace=True)

    X_train, preproc = preprocess(train.copy(), selected_features, is_train=True)
    X_test, _ = preprocess(
        test.copy(), selected_features, is_train=False, ref_pipeline=preproc
    )

    model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X_train, y)
    preds = model.predict(X_test)

    if use_cv:
        acc = evaluate_model(model, X_train, y, model_name="xgb")
    else:
        acc = None

    suffix = "_".join(map(str, feature_nums)) if feature_nums else "base"

    out_dir = f"submissions/2_xgb"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{out_dir}/submission_xgb_{suffix}.csv"
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds}).to_csv(
        out_file, index=False
    )

    print(f"✅ Saved predictions to {out_file}")
    if acc is not None:
        log_results("xgb", selected_features, acc, out_file)


def run_combinations():
    from modules.combination import XGB_COMBINATIONS

    for combo in XGB_COMBINATIONS:
        run_model(combo)


def run_all_general_combinations():
    from modules.combination import GENERAL_FEATURE_COMBINATIONS

    for combo in GENERAL_FEATURE_COMBINATIONS:
        run_model(combo)


def run_all_single_feature():
    for i in FEATURE_MAP.keys():
        run_model([i], use_cv=True)

    run_model([], use_cv=True)


if __name__ == "__main__":
    run_combinations()
    # run_all_single_feature()
    # run_all_general_combinations()
