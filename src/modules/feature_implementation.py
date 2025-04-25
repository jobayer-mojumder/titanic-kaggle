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
    5: "fare_group",
    6: "deck",
    7: "has_cabin",
    8: "ticket_prefix",
    9: "sex_pclass",
    10: "is_child",
    11: "women_children_first",
}

FEATURE_META = {
    "Title": {"type": "categorical"},
    "FamilySize": {"type": "numeric"},
    "IsAlone": {"type": "numeric"},
    "AgeGroup": {"type": "categorical"},
    "FareGroup": {"type": "categorical"},
    "Deck": {"type": "categorical"},
    "HasCabin": {"type": "numeric"},
    "TicketPrefix": {"type": "categorical"},
    "SexPclass": {"type": "numeric"},
    "IsChild": {"type": "numeric"},
    "WomenChildrenFirst": {"type": "numeric"},
}


# Shared helper
def compute_family_size(df):
    return df["SibSp"] + df["Parch"] + 1


def extract_deck(cabin):
    if pd.isna(cabin) or cabin == "":
        return "U"  # Unknown
    else:
        return cabin[0]


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
    df["FamilySize"] = family_size
    scaler = StandardScaler()
    df["FamilySize"] = scaler.fit_transform(df[["FamilySize"]])  # Normalize FamilySize
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


def add_fare_group(df):
    df["FarePerPerson"] = df["Fare"] / compute_family_size(df)
    df["FareGroup"] = pd.cut(
        df["FarePerPerson"],
        bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        labels=[
            "VeryLow",
            "Low",
            "Medium",
            "High",
            "VeryHigh",
            "UltraHigh",
            "Luxury",
            "UltraLuxury",
            "SuperLuxury",
            "MegaLuxury",
        ],
        right=True,
        include_lowest=True,
    )
    drop_columns = ["FarePerPerson"]
    df.drop(columns=drop_columns, inplace=True)
    return df


def add_deck(df):
    df["Deck"] = df["Cabin"].apply(extract_deck)
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
    "fare_group": add_fare_group,
    "deck": add_deck,
    "has_cabin": add_has_cabin,
    "ticket_prefix": add_ticket_prefix,
    "sex_pclass": add_sex_pclass,
    "is_child": add_is_child,
    "women_children_first": add_women_children_first,
}
