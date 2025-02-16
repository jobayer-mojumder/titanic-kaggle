import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")


# Basic preprocessing
def preprocess(df):
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked", "Pclass"])

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


# Prepare data
X = preprocess(train.drop("Survived", axis=1))
y = train["Survived"]

# Train model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# Prepare test predictions
X_test = preprocess(test)
test_predictions = model.predict(X_test)

# Create submission
pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_predictions}).to_csv(
    "dt_fe_2.csv", index=False
)
