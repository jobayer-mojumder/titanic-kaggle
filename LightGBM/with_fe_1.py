import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")


# Same preprocessing as previous models
def preprocess(df):
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
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

    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
    df["Title"] = df["Title"].replace(["Mme"], "Mrs")

    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Sex"], axis=1)
    df = pd.get_dummies(df, columns=["Embarked", "Pclass", "Title"])

    # Impute missing values (same strategy)
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


# Prepare data
X_train = preprocess(train.drop("Survived", axis=1))
y_train = train["Survived"]
X_test = preprocess(test)

# Train LightGBM model
model = LGBMClassifier(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Create submission
pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": model.predict(X_test)}
).to_csv("lgbm_fe_1.csv", index=False)
