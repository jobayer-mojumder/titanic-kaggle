# type: ignore
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")

# Add SexPclass feature before preprocessing
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})
train['SexPclass'] = train['Sex'] * train['Pclass']
test['SexPclass'] = test['Sex'] * test['Pclass']

# Function to extract Deck from Cabin
def get_deck(cabin):
    if pd.isna(cabin):
        return "Unknown"
    else:
        return cabin[0]

# Preprocessing function
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

    # Create IsMother feature: 1 for females with Parch > 0 and Age > 18
    df["IsMother"] = ((df["Sex"] == 1) & (df["Parch"] > 0) & (df["Age"] > 18)).astype(int)

    # Calculate FamilySize
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Create Deck feature
    df["Deck"] = df["Cabin"].apply(get_deck)
    
    # Drop unnecessary columns
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Age"], axis=1)
    
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=["Embarked", "Pclass", "Deck", "AgeGroup"])
    
    # If reference_columns is provided (for test set), align columns
    if reference_columns is not None:
        missing_cols = set(reference_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0  # Add missing columns with zeros
        df = df[reference_columns]  # Reorder to match training set
    
    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Prepare training data
X_train = preprocess(train.drop("Survived", axis=1))
y_train = train["Survived"]

# Prepare test data, aligning with training columns
X_test = preprocess(test, reference_columns=X_train.columns)

# Train LightGBM model
model = LGBMClassifier(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Create submission
pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": model.predict(X_test)}
).to_csv("submission_lgbm_comb_1.csv", index=False)

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})
print(feature_importance.sort_values(by="Importance", ascending=False))