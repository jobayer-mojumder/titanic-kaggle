# type: ignore
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")

# Preprocessing function
def preprocess(df):
    # Create HasCabin feature
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    
    # Drop unnecessary columns
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Sex"], axis=1)
    
    # Define numerical and categorical features
    numerical_features = ["Age", "Fare", "SibSp", "Parch", "HasCabin"]
    categorical_features = ["Pclass", "Embarked"]
    
    # Impute numerical features
    num_imputer = SimpleImputer(strategy="median")
    df[numerical_features] = num_imputer.fit_transform(df[numerical_features])
    
    # Impute categorical features
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df["Embarked"] = cat_imputer.fit_transform(df[["Embarked"]]).ravel()
    
    return df

# Prepare data
train_processed = preprocess(train)
X_train = train_processed.drop("Survived", axis=1)
y_train = train_processed["Survived"]
X_test = preprocess(test)

# Specify categorical features
categorical_features = ["Pclass", "Embarked"]

# Train CatBoost model
model = CatBoostClassifier(
    iterations=100, depth=3, random_seed=42, verbose=0, cat_features=categorical_features
)
model.fit(X_train, y_train)

# Create submission
pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": model.predict(X_test)}
).to_csv("submission_catboost_hascabin.csv", index=False)

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.get_feature_importance()
})
print(feature_importance.sort_values(by="Importance", ascending=False))