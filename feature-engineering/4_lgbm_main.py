# type: ignore
import os
import warnings
import pandas as pd
from lightgbm import LGBMClassifier
from modules.preprocessing import preprocess
from modules.feature_implementation import FEATURE_MAP
from modules.summary import log_results
from modules.evaluation import evaluate_model

warnings.filterwarnings("ignore", category=FutureWarning)


def run_model(feature_nums, use_cv=True):
    selected_features = [FEATURE_MAP[n] for n in feature_nums]
    print(f"🚀 Running lgbm with: {selected_features or 'Baseline only'}")

    train = pd.read_csv("../train.csv")
    test = pd.read_csv("../test.csv")

    y = train["Survived"]
    train.drop(columns=["Survived"], inplace=True)

    X_train, preproc = preprocess(train.copy(), selected_features, is_train=True)
    X_test, _ = preprocess(
        test.copy(), selected_features, is_train=False, ref_pipeline=preproc
    )

    model = LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=-1)
    model.fit(X_train, y)
    preds = model.predict(X_test)

    if use_cv:
        acc = evaluate_model(model, X_train, y, model_name="lgbm")
    else:
        acc = None

    suffix = "_".join(map(str, feature_nums)) if feature_nums else "base"

    out_dir = f"submissions/4_lgbm"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{out_dir}/submission_lgbm_{suffix}.csv"
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds}).to_csv(
        out_file, index=False
    )

    print(f"✅ Saved predictions to {out_file}")
    if acc is not None:
        log_results("lgbm", selected_features, acc, out_file)


def run_all_combinations():
    from modules.combination import LGBM_COMBINATIONS

    for combo in LGBM_COMBINATIONS:
        run_model(combo)


def run_all_single_feature():
    for i in FEATURE_MAP.keys():
        run_model([i], use_cv=True)

    run_model([], use_cv=True)


if __name__ == "__main__":
    # run_all_combinations()
    run_all_single_feature()
