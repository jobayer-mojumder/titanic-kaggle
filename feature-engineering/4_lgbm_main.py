import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SELECTED_FEATURES = []

# ------------------ Feature Engineering Functions ------------------

def add_title(df):
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\\.", expand=False)
    df["Title"] = df["Title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare"
    )
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return df

def add_family_size(df):
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    if "family_size" not in SELECTED_FEATURES:
        df.drop(columns=["FamilySize"], inplace=True)
    return df

def add_is_alone(df):
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df.drop(columns=["FamilySize"], inplace=True)
    return df

def add_age_group(df):
    def group(age):
        if pd.isna(age): return "Unknown"
        elif age <= 12: return "Child"
        elif age <= 60: return "Adult"
        else: return "Senior"
    df["AgeGroup"] = df["Age"].apply(group)
    return df

def add_fare_per_person(df):
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    df.drop(columns=["FamilySize"], inplace=True)
    return df

def add_deck(df):
    df["Deck"] = df["Cabin"].apply(lambda x: x[0] if pd.notna(x) else "Unknown")
    return df

def add_has_cabin(df):
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    return df

def add_is_mother(df):
    df["IsMother"] = ((df["Sex"] == "female") & (df["Parch"] > 0) & (df["Age"] > 18)).astype(int)
    return df

def add_sex_pclass(df):
    df["Sex"] = df["Sex"].map({"female": 1, "male": 0})
    df["SexPclass"] = df["Sex"] * df["Pclass"]
    return df

def add_is_child(df):
    df["IsChild"] = (df["Age"] <= 12).astype(int)
    return df

def add_women_children_first(df):
    df["WomenChildrenFirst"] = ((df["Sex"] == "female") | (df["Age"] <= 12)).astype(int)
    return df

# ------------------ Feature Registry ------------------

FEATURE_FUNCTIONS = {
    "title": add_title,
    "family_size": add_family_size,
    "is_alone": add_is_alone,
    "age_group": add_age_group,
    "fare_per_person": add_fare_per_person,
    "deck": add_deck,
    "has_cabin": add_has_cabin,
    "is_mother": add_is_mother,
    "sex_pclass": add_sex_pclass,
    "is_child": add_is_child,
    "women_children_first": add_women_children_first,
}

FEATURE_MAP = {
    1: "title", 2: "family_size", 3: "is_alone", 4: "age_group",
    5: "fare_per_person", 6: "deck", 7: "has_cabin", 8: "is_mother",
    9: "sex_pclass", 10: "is_child", 11: "women_children_first"
}

# ------------------ Preprocessing ------------------

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
    print(f"💡 Running LightGBM with: {SELECTED_FEATURES or 'Baseline only'}")

    train = pd.read_csv("../train.csv")
    test = pd.read_csv("../test.csv")

    y = train["Survived"]
    train = train.drop(columns=["Survived"])

    X_train, preproc = preprocess(train.copy(), SELECTED_FEATURES, is_train=True)
    X_test, _ = preprocess(test.copy(), SELECTED_FEATURES, is_train=False, ref_columns=preproc)

    model = LGBMClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y)

    preds = model.predict(X_test)
    suffix = "base" if not feature_nums else "_".join(map(str, feature_nums))
    out_file = f"submissions/4_LightGBM/submission_lgbm_{suffix}.csv"

    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds}).to_csv(out_file, index=False)
    print(f"✅ Saved predictions to {out_file}")

# ------------------ Main ------------------

if __name__ == "__main__":
    run_lgbm([9])  # [] → baseline features only, same as without_fe.py
