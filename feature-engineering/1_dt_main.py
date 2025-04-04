import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

# ------------------ Feature Engineering Functions ------------------

def add_title(df):
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\\.", expand=False)
    df["Title"] = df["Title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return df

def add_family_size(df):
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df

def add_is_alone(df):
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df.drop(columns=["FamilySize"], inplace=True)
    return df

def add_age_group(df):
    def categorize_age(age):
        if pd.isna(age): return "Unknown"
        elif age <= 12: return "Child"
        elif age <= 60: return "Adult"
        else: return "Senior"
    df["AgeGroup"] = df["Age"].apply(categorize_age)
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
    df["SexPclass"] = (df["Sex"].map({"female": 1, "male": 0}) * df["Pclass"]).astype(int)
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
    "women_children_first": add_women_children_first
}

FEATURE_MAP = {
    1: "title",
    2: "family_size",
    3: "is_alone",
    4: "age_group",
    5: "fare_per_person",
    6: "deck",
    7: "has_cabin",
    8: "is_mother",
    9: "sex_pclass",
    10: "is_child",
    11: "women_children_first"
}

# ------------------ Preprocessing ------------------

def preprocess(df, feature_names, is_train=True, reference_columns=None):
    for f in feature_names:
        df = FEATURE_FUNCTIONS[f](df)

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, errors="ignore")
    df = pd.get_dummies(df)

    if not is_train:
        for col in reference_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[reference_columns]

    return df

# ------------------ Runner ------------------

def run(features_by_number):
    feature_names = [FEATURE_MAP[n] for n in features_by_number]
    print(f"🚀 Running with features: {feature_names}")

    train = pd.read_csv("../train.csv")
    test = pd.read_csv("../test.csv")

    X = preprocess(train.drop("Survived", axis=1), feature_names, is_train=True)
    y = train["Survived"]
    X_test = preprocess(test.copy(), feature_names, is_train=False, reference_columns=X.columns)

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    preds = model.predict(X_test)

    filename = "submissions/1_DecisionTree/submission_dt_" + "_".join(map(str, features_by_number)) + ".csv"
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds}).to_csv(filename, index=False)
    print(f"✅ Saved to {filename}")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\n📊 Feature Importances:")
    print(importance)

# ------------------ Main ------------------

if __name__ == "__main__":
    run([])  # 👈 Change this list to try other combinations
