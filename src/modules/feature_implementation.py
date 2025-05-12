# type: ignore
import pandas as pd
from sklearn.preprocessing import StandardScaler
from modules.constant import MAX_FEATURES_NUM
import re

FEATURE_MAP = {
    1: "title",
    2: "family_size",
    3: "is_alone",
    4: "age_group",
    5: "fare_per_person",
    6: "sex_pclass",
    7: "deck",
    8: "is_mother",
    9: "is_child",
    10: "women_children_first",
    11: "ticket_prefix",
    12: "has_cabin",
}

FEATURE_META = {
    "Title": {"type": "categorical"},
    "FamilySize": {"type": "numeric"},
    "IsAlone": {"type": "numeric"},
    "AgeGroup": {"type": "categorical"},
    "FarePerPerson": {"type": "numeric"},
    "Deck": {"type": "categorical"},
    "IsMother": {"type": "numeric"},
    "HasCabin": {"type": "numeric"},
    "SexPclass": {"type": "numeric"},
    "IsChild": {"type": "numeric"},
    "WomenChildrenFirst": {"type": "numeric"},
    "TicketPrefix": {"type": "categorical"},
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
    df["FamilySize"] = compute_family_size(df)
    scaler = StandardScaler()
    df["FamilySize"] = scaler.fit_transform(df[["FamilySize"]])
    return df


def add_is_alone(df):
    df["IsAlone"] = (df["SibSp"] + df["Parch"] == 0).astype(int)
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
    family_size = compute_family_size(df).clip(lower=1)  # avoid division by zero
    fare_per_person = df["Fare"] / family_size
    fare_per_person = fare_per_person.fillna(0)  # fill NaN with 0
    df["FarePerPerson"] = fare_per_person.astype(int)  # safe to cast now
    return df


def add_deck(df):
    deck_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    df["Deck"] = (
        df["Cabin"].fillna("U0").map(lambda x: re.search(r"([A-Za-z]+)", x).group())
    )
    df["Deck"] = df["Deck"].map(deck_map).fillna(0).astype(int)
    return df


def add_is_mother(df):
    temp_added = False
    if "Title" not in df.columns:
        df = add_title(df)
        temp_added = True

    df["IsMother"] = (
        (df["Sex"] == 1)  # female
        & (df["Parch"] > 0)
        & (df["Title"] == "Mrs")
        & (df["Age"] >= 18)
    ).astype(int)

    if temp_added:
        df.drop(columns=["Title"], inplace=True)

    return df


def add_has_cabin(df):
    df["HasCabin"] = df["Cabin"].notnull().astype(int)
    return df


def add_ticket_prefix(df):
    def get_prefix(ticket):
        match = re.match(r"([A-Za-z./]+)", ticket)
        return match.group(1).strip().upper() if match else "NONE"

    df["TicketPrefix"] = df["Ticket"].apply(get_prefix)
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
    "is_mother": add_is_mother,
    "has_cabin": add_has_cabin,
    "sex_pclass": add_sex_pclass,
    "is_child": add_is_child,
    "women_children_first": add_women_children_first,
    "ticket_prefix": add_ticket_prefix,
}
