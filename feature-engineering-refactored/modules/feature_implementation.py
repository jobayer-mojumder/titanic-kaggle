# type: ignore
import pandas as pd


def add_title(df):
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\\.", expand=False)
    df["Title"] = df["Title"].replace(
        [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ],
        "Rare",
    )
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
        if pd.isna(age):
            return "Unknown"
        elif age <= 12:
            return "Child"
        elif age <= 60:
            return "Adult"
        else:
            return "Senior"

    df["AgeGroup"] = df["Age"].apply(categorize_age)
    df.drop(columns=["Age"], inplace=True)
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
    df["IsMother"] = (
        (df["Sex"] == "female") & (df["Parch"] > 0) & (df["Age"] > 18)
    ).astype(int)
    return df


def add_sex_pclass(df):
    if "Sex" not in df.columns or "Pclass" not in df.columns:
        return df

    df["SexMapped"] = df["Sex"].map({"female": 1, "male": 0})
    df["Pclass"] = pd.to_numeric(df["Pclass"], errors="coerce")
    df["SexPclass"] = (df["SexMapped"] * df["Pclass"]).fillna(0).astype(int)

    df.drop(columns=["SexMapped"], inplace=True)
    return df


def add_is_child(df):
    df["IsChild"] = (df["Age"] <= 12).astype(int)
    return df


def add_women_children_first(df):
    df["WomenChildrenFirst"] = ((df["Sex"] == "female") | (df["Age"] <= 12)).astype(int)
    return df


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
    11: "women_children_first",
}

FEATURE_META = {
    "title": {"type": "categorical"},
    "family_size": {"type": "numeric"},
    "is_alone": {"type": "numeric"},
    "age_group": {"type": "categorical"},
    "fare_per_person": {"type": "numeric"},
    "deck": {"type": "categorical"},
    "has_cabin": {"type": "numeric"},
    "is_mother": {"type": "numeric"},
    "sex_pclass": {"type": "numeric"},
    "is_child": {"type": "numeric"},
    "women_children_first": {"type": "numeric"},
}
