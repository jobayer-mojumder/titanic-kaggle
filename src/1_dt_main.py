# type: ignore
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from modules.preprocessing import preprocess
from modules.feature_implementation import FEATURE_MAP
from modules.summary import log_results
from modules.evaluation import evaluate_model
from modules.model_tuning import tune_model
from modules.constant import DEFAULT_MODELS


def run_model(feature_nums, use_cv=True, tune=False):
    selected_features = [FEATURE_MAP[n] for n in feature_nums]
    print(f"🚀 Running dt with: {selected_features or 'Baseline only'}")

    train = pd.read_csv("../train.csv")
    test = pd.read_csv("../test.csv")

    y = train["Survived"]
    train.drop(columns=["Survived"], inplace=True)

    X_train, preproc = preprocess(train.copy(), selected_features, is_train=True)
    X_test, _ = preprocess(
        test.copy(), selected_features, is_train=False, ref_pipeline=preproc
    )

    if tune:
        model = tune_model(X_train, y, model_key="dt")
    else:
        model = DEFAULT_MODELS["dt"]

    model.fit(X_train, y)
    preds = model.predict(X_test)

    if use_cv:
        acc = evaluate_model(model, X_train, y, model_name="dt")
    else:
        acc = None

    suffix = "_".join(map(str, feature_nums)) if feature_nums else "base"
    out_dir = f"submissions/1_dt"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{out_dir}/submission_dt_{suffix}.csv"
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds}).to_csv(
        out_file, index=False
    )

    print(f"✅ Saved predictions to {out_file}")
    if acc is not None:
        log_results("dt", selected_features, acc, out_file)


def run_combinations():
    from modules.combination import DT_COMBINATIONS

    for combo in DT_COMBINATIONS:
        run_model(combo)


def run_all_general_combinations():
    from modules.combination import GENERAL_FEATURE_COMBINATIONS

    for combo in GENERAL_FEATURE_COMBINATIONS:
        run_model(combo)


def run_all_single_feature():
    for i in FEATURE_MAP.keys():
        run_model([i], use_cv=True)

    run_model([], use_cv=True)


def run_baseline():
    run_model([], use_cv=True)


def run_baseline_tune():
    run_model([], use_cv=True, tune=True)


def run_all_single_feature_tune():
    for i in FEATURE_MAP.keys():
        run_model([i], use_cv=True, tune=True)


def run_combinations_tune():
    from modules.combination import DT_COMBINATIONS

    for combo in DT_COMBINATIONS:
        run_model(combo, tune=True)


def run_all_general_combinations_tune():
    from modules.combination import GENERAL_FEATURE_COMBINATIONS

    for combo in GENERAL_FEATURE_COMBINATIONS:
        run_model(combo, tune=True)


if __name__ == "__main__":
    # run_combinations()
    # run_all_single_feature()
    # run_all_general_combinations()
    # run_baseline()
    # run_baseline_tune()
    # run_all_single_feature_tune()
    # run_combinations_tune()
    run_all_general_combinations_tune()
