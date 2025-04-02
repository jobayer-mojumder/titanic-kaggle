# type: ignore
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")


# Updated preprocessing function to include AgeGroup
def preprocess(df, reference_columns=None):
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

    # Drop unnecessary columns (including original Age since AgeGroup replaces it)
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Age"], axis=1)

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # One-hot encode categorical columns: Embarked, Pclass, and AgeGroup
    df = pd.get_dummies(df, columns=["Embarked", "Pclass", "AgeGroup"])

    # Align columns to match training set
    if reference_columns is not None:
        missing_cols = set(reference_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        df = df[reference_columns]

    # Impute missing values for numerical columns
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


# Prepare data
X = preprocess(train.drop("Survived", axis=1))
y = train["Survived"]

# Train model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# Prepare test predictions
X_test = preprocess(test, reference_columns=X.columns)
test_predictions = model.predict(X_test)

# Create submission
pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_predictions}).to_csv(
    "submission_dt_agegroup.csv", index=False
)

# Optional: Feature importance (based on splits)
feature_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
)
print(feature_importance.sort_values(by="Importance", ascending=False))
