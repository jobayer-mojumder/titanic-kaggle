#type: ignore
import os
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from modules.feature_implementation import FEATURE_FUNCTIONS, FEATURE_MAP, SELECTED_FEATURES
from modules.combination import GENERAL_FEATURE_COMBINATIONS

def preprocess(df, features_to_use, is_train=True, ref_columns=None):
    if not features_to_use:
        features_to_use = []  # baseline case

    for f in features_to_use:
        df = FEATURE_FUNCTIONS[f](df)

    # Drop non-feature columns
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, errors="ignore")

    # Encode Sex as 0/1
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"female": 1, "male": 0})

    # Define baseline columns (always used if no features selected)
    if not SELECTED_FEATURES:
        numerical_features = ["Age", "SibSp", "Parch", "Fare", "Sex"]
        categorical_features = ["Pclass", "Embarked"]
    else:
        known_numeric = [
            "Age", "SibSp", "Parch", "Fare", "Sex", "SexPclass", "FarePerPerson",
            "FamilySize", "IsAlone", "IsChild", "IsMother", "WomenChildrenFirst", "HasCabin"
        ]
        numerical_features = [col for col in known_numeric if col in df.columns]
        categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), numerical_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features),
    ])

    if is_train:
        X = preprocessor.fit_transform(df)
        return X, preprocessor
    else:
        X = ref_columns.transform(df)
        return X, ref_columns

# ------------------ Runner ------------------

def run_xgboost(feature_nums):
    global SELECTED_FEATURES
    SELECTED_FEATURES = [FEATURE_MAP[n] for n in feature_nums]
    print(f"🚀 Running XGBoost with: {SELECTED_FEATURES or 'Baseline only'}")

    train = pd.read_csv("../train.csv")
    test = pd.read_csv("../test.csv")

    y = train["Survived"]
    train = train.drop(columns=["Survived"])

    X_train, preproc = preprocess(train.copy(), SELECTED_FEATURES, is_train=True)
    X_test, _ = preprocess(test.copy(), SELECTED_FEATURES, is_train=False, ref_columns=preproc)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train, y)

    preds = model.predict(X_test)

    output_dir = "submissions/2_XGBoost"
    os.makedirs(output_dir, exist_ok=True)

    suffix = "_".join(map(str, feature_nums)) if feature_nums else "base"
    out_file = f"{output_dir}/submission_xgb_{suffix}.csv"

    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds}).to_csv(out_file, index=False)
    print(f"✅ Saved predictions to {out_file}")

def run_all_single_features():
    for i in range(1, len(FEATURE_MAP) + 1):
        run_xgboost([i])
    run_xgboost([])

def run_general_combinations():
    # Run all general combinations
    for combination in GENERAL_FEATURE_COMBINATIONS:
        run_xgboost(combination)

# ------------------ Main ------------------

if __name__ == "__main__":
    # run_all_single_features()
    run_general_combinations()