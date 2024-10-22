# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (replace 'titanic.csv' with the path to your dataset)
df = pd.read_csv('Titanic-Dataset.csv')

# Display the first few rows of the dataset
print(df.head())

# Data Preprocessing
# Drop unnecessary columns: PassengerId, Name, Ticket, and Cabin (these are not relevant for survival prediction)
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill missing values in 'Age' with the median value
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing values in 'Embarked' with the most common value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical variables ('Sex' and 'Embarked')
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])  # 0 for female, 1 for male
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])  # Embarked: 0 = Cherbourg, 1 = Queenstown, 2 = Southampton

# Define feature matrix (X) and target variable (y)
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print detailed classification report
print(classification_report(y_test, y_pred))

# Feature Importance (to understand which factors were most important)
plt.figure(figsize=(10, 6))
sns.barplot(x=clf.feature_importances_, y=X.columns)
plt.title('Feature Importance')
plt.show()

# User Input for Survival Prediction
def get_user_input():
    print("Enter details for survival prediction:")
    
    # Take user inputs
    Pclass = int(input("Enter Pclass (1 = First, 2 = Second, 3 = Third): "))
    Sex = input("Enter Sex (male/female): ").strip().lower()
    if Sex == 'male':
        Sex = 1
    else:
        Sex = 0
    Age = float(input("Enter Age: "))
    SibSp = int(input("Enter number of Siblings/Spouses aboard (SibSp): "))
    Parch = int(input("Enter number of Parents/Children aboard (Parch): "))
    Fare = float(input("Enter Fare paid: "))
    Embarked = input("Enter Embarked (C = Cherbourg, Q = Queenstown, S = Southampton): ").strip().upper()
    
    if Embarked == 'C':
        Embarked = 0
    elif Embarked == 'Q':
        Embarked = 1
    else:
        Embarked = 2
    
    # Create a dictionary of the user input with proper column names
    user_data = {
        'Pclass': [Pclass],
        'Sex': [Sex],
        'Age': [Age],
        'SibSp': [SibSp],
        'Parch': [Parch],
        'Fare': [Fare],
        'Embarked': [Embarked]
    }
    
    # Convert the dictionary to a DataFrame (so it has feature names)
    return pd.DataFrame(user_data)

# Get user input
person_data = get_user_input()

# Predict survival for the new individual
prediction = clf.predict(person_data)

# Print the prediction result
if prediction[0] == 1:
    print("The person is predicted to survive.")
else:
    print("The person is predicted to not survive.")
