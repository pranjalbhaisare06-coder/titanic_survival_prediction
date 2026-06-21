
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

def train_and_save_model(data_path='train.csv', model_output_path='model.pkl'):
    # Load the data
    titanic_data = pd.read_csv(data_path)

    # Drop 'Cabin' column
    titanic_data = titanic_data.drop(columns='Cabin', axis=1)

    # Fill missing 'Age' with mean
    titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

    # Fill missing 'Embarked' with mode
    titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

    # Encode categorical features
    titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

    # Define features (X) and target (Y)
    X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
    Y = titanic_data['Survived']

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
    model.fit(X_train, Y_train)

    # Save the trained model
    with open(model_output_path, 'wb') as file:
        pickle.dump(model, file)

    print(f"Model trained and saved as '{model_output_path}' successfully!")

if __name__ == '__main__':
    train_and_save_model()
