import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/train.csv")


# def plot_survival_count(df):
#     survival_counts = df["Survived"].value_counts()

#     plt.figure(figsize=(8, 5))
#     sns.barplot(x=survival_counts.index, y=survival_counts.values, palette="viridis")
#     plt.title("Survival Count")
#     plt.xlabel("Survived (0 = No, 1 = Yes)")
#     plt.ylabel("Number of Passengers")
#     plt.xticks(ticks=[0, 1], labels=["No", "Yes"])
#     plt.grid(axis="y")
#     plt.show()


# plot_survival_count(df)


# def plot_age_distribution(df):
#     plt.figure(figsize=(10, 6))
#     sns.histplot(df["Age"].dropna(), bins=30, kde=True, color="blue")
#     plt.title("Age Distribution of Passengers")
#     plt.xlabel("Age")
#     plt.ylabel("Frequency")
#     plt.grid(axis="y")
#     plt.show()


# plot_age_distribution(df)


# def plot_class_distribution(df):
#     class_counts = df["Pclass"].value_counts()

#     plt.figure(figsize=(8, 5))
#     sns.barplot(x=class_counts.index, y=class_counts.values, palette="coolwarm")
#     plt.title("Passenger Class Distribution")
#     plt.xlabel("Passenger Class")
#     plt.ylabel("Number of Passengers")
#     plt.xticks(ticks=[0, 1, 2], labels=["1st Class", "2nd Class", "3rd Class"])
#     plt.grid(axis="y")
#     plt.show()


# plot_class_distribution(df)


# def plot_survival_by_class(df):
#     plt.figure(figsize=(10, 6))
#     sns.countplot(x="Pclass", hue="Survived", data=df, palette="Set1")
#     plt.title("Survival Count by Passenger Class")
#     plt.xlabel("Passenger Class")
#     plt.ylabel("Number of Passengers")
#     plt.xticks(ticks=[0, 1, 2], labels=["1st Class", "2nd Class", "3rd Class"])
#     plt.legend(title="Survived", loc="upper right", labels=["No", "Yes"])
#     plt.grid(axis="y")
#     plt.show()


# plot_survival_by_class(df)


# def plot_survival_count_by_sex(df):
#     plt.figure(figsize=(10, 6))
#     sns.countplot(x="Sex", hue="Survived", data=df, palette="Set1")
#     plt.title("Survival Count by Sex")
#     plt.xlabel("Sex")
#     plt.ylabel("Number of Passengers")
#     plt.xticks(ticks=[0, 1], labels=["Female", "Male"])
#     plt.legend(title="Survived", loc="upper right", labels=["No", "Yes"])
#     plt.grid(axis="y")
#     plt.show()


# plot_survival_count_by_sex(df)


# def plot_survival_by_embarked(df):
#     plt.figure(figsize=(10, 6))
#     sns.countplot(x="Embarked", hue="Survived", data=df, palette="Set1")
#     plt.title("Survival Count by Embarked Port")
#     plt.xlabel("Embarked Port")
#     plt.ylabel("Number of Passengers")
#     plt.xticks(
#         ticks=[0, 1, 2], labels=["C = Cherbourg", "Q = Queenstown", "S = Southampton"]
#     )
#     plt.legend(title="Survived", loc="upper right", labels=["No", "Yes"])
#     plt.grid(axis="y")
#     plt.show()


# plot_survival_by_embarked(df)


# missing values for each column in number and percentage format as a bar plot. Also need to show the number of missing values in the bar plot.


def plot_missing_values(df):
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100

    plt.figure(figsize=(12, 6))
    sns.barplot(x=missing_counts.index, y=missing_counts.values, palette="viridis")
    plt.title("Missing Values in Each Column")
    plt.xlabel("Columns")
    plt.ylabel("Number of Missing Values")

    for i, v in enumerate(missing_counts.values):
        plt.text(
            i, v + 5, f"{v} ({missing_percentages[i]:.1f}%)", ha="center", va="bottom"
        )

    plt.xticks(rotation=45)
    plt.grid(axis="y")
    plt.show()


plot_missing_values(df)
