# type: ignore
import pandas as pd

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
    "Title": {"type": "categorical"},
    "FamilySize": {"type": "categorical"},
    "IsAlone": {"type": "numeric"},
    "AgeGroup": {"type": "categorical"},
    "FarePerPerson": {"type": "numeric"},
    "Deck": {"type": "categorical"},
    "HasCabin": {"type": "numeric"},
    "IsMother": {"type": "numeric"},
    "SexPclass": {"type": "numeric"},
    "IsChild": {"type": "numeric"},
    "WomenChildrenFirst": {"type": "numeric"},
}


# Shared helper
def compute_family_size(df):
    return df["SibSp"] + df["Parch"] + 1


# Feature functions
def add_title(df):
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\\.", expand=False)
    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
    df["Title"] = df["Title"].replace("Mme", "Mrs")
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
    return df


def add_family_size(df):
    family_size = compute_family_size(df)
    df["FamilySize"] = pd.cut(
        family_size,
        bins=[0, 1, 4, 6, float("inf")],
        labels=["Alone", "Small", "Medium", "Large"],
        right=True,
        include_lowest=True,
    )
    return df


def add_is_alone(df):
    df["IsAlone"] = (compute_family_size(df) == 1).astype(int)
    return df


def add_age_group(df):
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 60, 120],
        labels=["Child", "Adult", "Senior"],
        right=True,
        include_lowest=True,
    )
    return df


def add_fare_per_person(df):
    df["FarePerPerson"] = df["Fare"] / compute_family_size(df)
    return df


def add_deck(df):
    df["Deck"] = df["Cabin"].fillna("U").map(lambda x: x[0])
    return df


def add_has_cabin(df):
    df["HasCabin"] = df["Cabin"].notnull().astype(int)
    return df


def add_is_mother(df):
    df["IsMother"] = ((df["Sex"] == 1) & (df["Parch"] >= 1) & (df["Age"] >= 18)).astype(
        int
    )
    return df


def add_sex_pclass(df):
    df["SexPclass"] = df["Sex"] * df["Pclass"].astype(int)
    return df


def add_is_child(df):
    df["IsChild"] = (df["Age"] <= 12).astype(int)
    return df


def add_women_children_first(df):
    df["WomenChildrenFirst"] = ((df["Sex"] == 1) | (df["Age"] <= 12)).astype(int)
    return df


# Register all features
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
