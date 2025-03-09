# type: ignore

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")


# Updated preprocessing function to include AgeGroup
def preprocess(df):
    # Create AgeGroup feature before dropping or encoding
    def categorize_age(age):
        if pd.isna(age):
            return "Unknown"  # Handle missing values
        elif age <= 12:
            return "Child"
        elif age <= 60:
            return "Adult"
        else:
            return "Senior"

    df["AgeGroup"] = df["Age"].apply(categorize_age)

    # Drop unnecessary columns
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Sex", "Age"], axis=1)

    df = pd.get_dummies(df, columns=["Embarked", "Pclass"])

    # Impute missing values for numerical columns
    imputer = SimpleImputer(strategy="median")
    df_numeric = pd.DataFrame(
        imputer.fit_transform(df.drop("AgeGroup", axis=1)),
        columns=df.drop("AgeGroup", axis=1).columns,
    )

    # Combine numeric imputed data with AgeGroup
    df_final = pd.concat([df_numeric, df[["AgeGroup"]].reset_index(drop=True)], axis=1)

    return df_final


# Prepare data
X_train = preprocess(train.drop("Survived", axis=1))
y_train = train["Survived"]
X_test = preprocess(test)

# Specify categorical features for CatBoost
categorical_features = ["AgeGroup"]

# Train CatBoost model
model = CatBoostClassifier(
    iterations=100,
    depth=3,
    random_seed=42,
    verbose=0,  # Silent mode
    cat_features=categorical_features,  # Tell CatBoost which features are categorical
)
model.fit(X_train, y_train)

# Create submission
pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": model.predict(X_test)}
).to_csv("submission_catboost.csv", index=False)

# Optional: Check feature importance
feature_importance = pd.DataFrame(
    {"Feature": X_train.columns, "Importance": model.get_feature_importance()}
)
print(feature_importance.sort_values(by="Importance", ascending=False))
