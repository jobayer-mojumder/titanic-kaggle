# type: ignore
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")

# Feature Engineering: Create IsAlone feature
def create_is_alone(df):
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1  # Create FamilySize first
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)  # 1 if alone, else 0
    df = df.drop(["FamilySize"], axis=1)  # Drop FamilySize if not needed
    return df


# Apply feature engineering to both train and test data
train = create_is_alone(train)
test = create_is_alone(test)

# Function to extract Deck from Cabin
def get_deck(cabin):
    if pd.isna(cabin):
        return "Unknown"
    else:
        return cabin[0]

# Preprocessing function
def preprocess(df, reference_columns=None):
    # Create Deck feature
    df["Deck"] = df["Cabin"].apply(get_deck)

    df["IsMother"] = ((df["Sex"] == "female") & (df["Parch"] > 0) & (df["Age"] > 18)).astype(int)
    
    # Drop unnecessary columns
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=["Embarked", "Pclass", "Deck"])
    
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
X = preprocess(train.drop("Survived", axis=1))
y = train["Survived"]

# Prepare test data, aligning with training columns
X_test = preprocess(test, reference_columns=X.columns)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create submission
pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": model.predict(X_test)}
).to_csv("submission_rf_comb_2.csv", index=False)

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})
print(feature_importance.sort_values(by="Importance", ascending=False))