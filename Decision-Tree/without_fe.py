import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

# Basic preprocessing (same as previous versions)
def preprocess(df):
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked', 'Pclass'])
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Prepare data
X_train = preprocess(train.drop('Survived', axis=1))
y_train = train['Survived']
X_test = preprocess(test)

# Train model
model = DecisionTreeClassifier(max_depth=3, random_state=42)  # Constrained depth to prevent overfitting
model.fit(X_train, y_train)

# Create submission
pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': model.predict(X_test)
}).to_csv('submission_dt.csv', index=False)