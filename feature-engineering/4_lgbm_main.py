#type: ignore
import os
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from modules.feature_implementation import FEATURE_FUNCTIONS, FEATURE_MAP, SELECTED_FEATURES

def preprocess(df, features_to_use, is_train=True, ref_columns=None):
    if not features_to_use:
        features_to_use = []

    for f in features_to_use:
        df = FEATURE_FUNCTIONS[f](df)

    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, errors="ignore")

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"female": 1, "male": 0})
    if "Pclass" in df.columns:
        df["Pclass"] = df["Pclass"].astype(str)

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

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numerical_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

    if is_train:
        X = preprocessor.fit_transform(df)
        return X, preprocessor
    else:
        X = ref_columns.transform(df)
        return X, ref_columns

# ------------------ Runner ------------------

def run_lgbm(feature_nums):
    global SELECTED_FEATURES
    SELECTED_FEATURES = [FEATURE_MAP[n] for n in feature_nums]
    print(f"🚀 Running LightGBM with: {SELECTED_FEATURES or 'Baseline only'}")

    train = pd.read_csv("../train.csv")
    test = pd.read_csv("../test.csv")

    y = train["Survived"]
    train = train.drop(columns=["Survived"])

    X_train, preproc = preprocess(train.copy(), SELECTED_FEATURES, is_train=True)
    X_test, _ = preprocess(test.copy(), SELECTED_FEATURES, is_train=False, ref_columns=preproc)

    model = LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=-1)
    model.fit(X_train, y)

    preds = model.predict(X_test)

    # Ensure output directory exists
    output_dir = "submissions/4_LightGBM"
    os.makedirs(output_dir, exist_ok=True)

    suffix = "base" if not feature_nums else "_".join(map(str, feature_nums))
    out_file = f"{output_dir}/submission_lgbm_{suffix}.csv"

    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds}).to_csv(out_file, index=False)
    print(f"✅ Saved predictions to {out_file}")


# ------------------ Main ------------------

if __name__ == "__main__":
    run_lgbm([9])  # [] → baseline features only, same as without_fe.py
