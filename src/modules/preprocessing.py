# type: ignore
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from modules.feature_implementation import FEATURE_FUNCTIONS, FEATURE_META

ALWAYS_INCLUDED_NUMERIC = ["Sex"]
ALWAYS_INCLUDED_CATEGORICAL = ["Pclass", "Embarked"]
BASELINE_FEATURES = ["Sex", "Pclass", "Embarked", "Age", "SibSp", "Parch", "Fare"]


def apply_features(df, feature_names):
    for name in feature_names:
        df = FEATURE_FUNCTIONS[name](df)
    return df


def get_column_types(df, selected_features=None):
    known_numeric = [k for k, v in FEATURE_META.items() if v["type"] == "numeric"]

    if not selected_features:
        num_cols = ["Age", "SibSp", "Parch", "Fare", "Sex"]
        cat_cols = ["Pclass", "Embarked"]
    else:
        num_cols = list(
            set(ALWAYS_INCLUDED_NUMERIC + [f for f in known_numeric if f in df.columns])
        )
        cat_cols = list(
            set(
                ALWAYS_INCLUDED_CATEGORICAL
                + df.select_dtypes(include=["object", "category"]).columns.tolist()
            )
        )

    return num_cols, cat_cols


def build_pipeline(numerical, categorical, model_key=None):
    transformers = [("num", SimpleImputer(strategy="median"), numerical)]

    if model_key != "cb":
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    else:
        transformers.append(
            (
                "cat",
                SimpleImputer(strategy="most_frequent"),
                categorical,
            )
        )

    return ColumnTransformer(transformers=transformers)


def preprocess(df, feature_names, is_train=True, ref_pipeline=None, model_key=None):
    df = df.copy()

    # 🧼 Data cleaning
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"female": 1, "male": 0})
    if "Pclass" in df.columns:
        df["Pclass"] = df["Pclass"].astype(str)
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna("S")

    # Apply engineered features
    df = apply_features(df, feature_names)

    # Drop irrelevant columns
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, errors="ignore")

    # Extract column types
    numeric, categorical = get_column_types(df, feature_names)

    # Build pipeline
    pipeline = build_pipeline(numeric, categorical, model_key)

    if is_train:
        X = pipeline.fit_transform(df)
        return X, pipeline
    else:
        X = ref_pipeline.transform(df)
        return X, ref_pipeline
