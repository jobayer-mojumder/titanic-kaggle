# type: ignore
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")

# Function to extract Deck from Cabin
def get_deck(cabin):
    if pd.isna(cabin):
        return "Unknown"
    else:
        return cabin[0]

# Preprocessing function
def preprocess(df):
    # Create Deck feature
    df["Deck"] = df["Cabin"].apply(get_deck)
    
    # Drop unnecessary columns
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Sex"], axis=1)
    
    # Define numerical and categorical features
    numerical_features = ["Age", "Fare", "SibSp", "Parch"]
    categorical_features = ["Pclass", "Embarked", "Deck"]
    
    # Impute numerical features with median
    num_imputer = SimpleImputer(strategy="median")
    df[numerical_features] = num_imputer.fit_transform(df[numerical_features])
    
    # Impute categorical features with most frequent, flatten to 1D
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df["Embarked"] = cat_imputer.fit_transform(df[["Embarked"]]).ravel()
    
    return df

# Prepare data
train_processed = preprocess(train)
X_train = train_processed.drop("Survived", axis=1)
y_train = train_processed["Survived"]
X_test = preprocess(test)

# Specify categorical features
categorical_features = ["Pclass", "Embarked", "Deck"]

# Train CatBoost model
model = CatBoostClassifier(
    iterations=100, depth=3, random_seed=42, verbose=0, cat_features=categorical_features
)
model.fit(X_train, y_train)

# Create submission
pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": model.predict(X_test)}
).to_csv("submission_catboost_deck.csv", index=False)

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.get_feature_importance()
})
print(feature_importance.sort_values(by="Importance", ascending=False))