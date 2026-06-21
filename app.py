
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle # Import pickle to load the model

app = Flask(__name__)

# --- Data Loading and Preprocessing (Replicated from Notebook) ---
def load_and_preprocess_data():
    # This function is used to get the original data structure and preprocessing logic
    # In a real deployment, you might load a preprocessed dataset or a pre-fitted preprocessor
    titanic_data = pd.read_csv('train.csv')

    # Drop 'Cabin' column
    titanic_data = titanic_data.drop(columns=['Cabin'], errors='ignore')

    # Fill missing 'Age' with mean
    titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

    # Fill missing 'Embarked' with mode
    titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

    # Encode categorical features
    titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
    
    # Select features (X) and target (Y)
    X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
    Y = titanic_data['Survived']
    
    return X, Y, titanic_data # Return titanic_data for consistency if needed elsewhere

# Load and preprocess data to get the feature columns (X) and original titanic_data for reference
X_train_ref, _, titanic_data_ref = load_and_preprocess_data()

# Load the pre-trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# --- Flask Routes ---
@app.route('/')
def home():
    return 'Titanic Survival Prediction App is running!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Prepare input data for prediction
    # Ensure the input data has the same columns and order as X_train_ref
    input_df = pd.DataFrame([data])
    
    # Apply the same preprocessing steps as training data
    # Drop columns not used in training (if they exist in input)
    cols_to_drop_if_present = ['PassengerId', 'Name', 'Ticket', 'Survived', 'Cabin']
    for col in cols_to_drop_if_present:
        if col in input_df.columns:
            input_df = input_df.drop(columns=col)

    # Handle 'Age' missing values
    if 'Age' in input_df.columns and input_df['Age'].isnull().any():
        # Use the mean from the training data for consistency
        input_df['Age'].fillna(X_train_ref['Age'].mean(), inplace=True)
        
    # Handle 'Embarked' missing values and encoding
    if 'Embarked' in input_df.columns:
        if input_df['Embarked'].isnull().any():
            # Use the mode from the original training data for consistency
            input_df['Embarked'].fillna(titanic_data_ref['Embarked'].mode()[0], inplace=True)
        input_df.replace({'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

    # Handle 'Sex' encoding
    if 'Sex' in input_df.columns:
        input_df.replace({'Sex':{'male':0,'female':1}}, inplace=True)

    # Ensure all expected columns are present, fill with 0 or mean if missing
    # This is a simplified approach; a more robust solution would use a ColumnTransformer
    # or similar preprocessor fit on the training data.
    for col in X_train_ref.columns:
        if col not in input_df.columns:
            # Fill missing columns with a default value (e.g., 0 or mean of training column)
            input_df[col] = 0 # Assuming 0 is a reasonable default for new/missing features

    # Reorder columns to match training data order
    input_df = input_df[X_train_ref.columns]

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    result = {
        'prediction': int(prediction[0]),
        'probability_not_survived': prediction_proba[0][0],
        'probability_survived': prediction_proba[0][1]
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
