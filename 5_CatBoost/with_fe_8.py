# type: ignore
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")

# Preprocessing function
def preprocess(df):
    # Create IsMother feature: 1 for females with Parch > 0 and Age > 18
    df["IsMother"] = ((df["Sex"] == "female") & (df["Parch"] > 0) & (df["Age"] > 18)).astype(int)
    
    # Drop unnecessary columns
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    
    df = pd.get_dummies(df, columns=["Embarked", "Pclass"])

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns) 

# Prepare data
X_train = preprocess(train.drop("Survived", axis=1))
y_train = train["Survived"]
X_test = preprocess(test)

# Train CatBoost model
model = CatBoostClassifier(
    iterations=100, depth=3, random_seed=42, verbose=0  # Silent mode
)
model.fit(X_train, y_train)

# Create submission
pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": model.predict(X_test)}
).to_csv("submission_catboost_isMother.csv", index=False)

# Optional: Feature importance
feature_importance = pd.DataFrame(
    {"Feature": X_train.columns, "Importance": model.get_feature_importance()}
)
print(feature_importance.sort_values(by="Importance", ascending=False))