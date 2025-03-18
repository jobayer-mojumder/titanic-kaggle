# type: ignore
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")


# Updated preprocessing function to include AgeGroup
def preprocess(df):
    # Create AgeGroup feature
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

    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Age"], axis=1)

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # One-hot encode categorical columns: Embarked, Pclass, and AgeGroup
    df = pd.get_dummies(df, columns=["Embarked", "Pclass", "AgeGroup"])

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


# Prepare data
X = preprocess(train.drop("Survived", axis=1))
y = train["Survived"]
X_test = preprocess(test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create submission
pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": model.predict(X_test)}
).to_csv("submission_rf_agegroup.csv", index=False)

# Optional: Feature importance
feature_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
)
print(feature_importance.sort_values(by="Importance", ascending=False))
