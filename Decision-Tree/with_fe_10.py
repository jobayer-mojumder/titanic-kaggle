# type: ignore
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")

# Preprocessing function
def preprocess(df, reference_columns=None):
    # Drop unnecessary columns, keeping Embarked as the feature of interest
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Sex"], axis=1)
    
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=["Embarked", "Pclass"])
    
    # Align test set columns with training set
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
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# Create submission
pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": model.predict(X_test)}
).to_csv("submission_dt_embarked.csv", index=False)

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})
print(feature_importance.sort_values(by="Importance", ascending=False))