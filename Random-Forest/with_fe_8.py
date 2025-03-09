# type: ignore
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")

# Preprocessing function
def preprocess(df, reference_columns=None):
    # Create HasCabin feature
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    
    # Drop unnecessary columns
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Sex"], axis=1)
    
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=["Embarked", "Pclass"])
    
    # Align test set columns with training set if reference_columns provided
    if reference_columns is not None:
        missing_cols = set(reference_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        df = df[reference_columns]
    
    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Prepare training data
X = preprocess(train.drop("Survived", axis=1))
y = train["Survived"]

# Prepare test data
X_test = preprocess(test, reference_columns=X.columns)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create submission
pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": model.predict(X_test)}
).to_csv("submission_rf_hascabin.csv", index=False)

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})
print(feature_importance.sort_values(by="Importance", ascending=False))