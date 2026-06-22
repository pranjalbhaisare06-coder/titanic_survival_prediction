## Abstract
Titanic Survival Prediction This project uses Machine Learning technique to predict the survival of a passenger on the Titanic. It uses features like age, gender, passenger class, fare and family details for prediction. The project includes data preprocessing, handling missing values, encoding features, training a model, and making predictions. We have used a classification algorithm called Logistic Regression on the passenger data to estimate the survival chances. The system we built demonstrates how Machine Learning can be applied with historical datasets to do predictive analysis and decision making.

## introduction
Machine Learning has become an important tool for analysis and predictions of historical data. Titanic disaster dataset is one of the most popular dataset to understand classification problems. The aim of this project is to develop a predictive model to predict if a passenger would survive based on personal and travel related information. The system finds what factors played a role in survival by looking for patterns in the data and predicts for new passenger records.

## Literature review
The Titanic dataset has been used in several studies to predict the survival of passengers using Machine Learning algorithms. Researchers have used algorithms such as Logistic Regression, Decision Trees, Random Forest, Support Vector Machines and Neural Networks. Logistic Regression is widely utilised for binary classification tasks due to its simplicity and effectiveness. Previous works have shown that features such as gender, passenger class and age have a high influence on the probability of survival. These researches provide a basis for building accurate prediction models based on historical passenger data.at influenced survival predictions and new passenger records.

## Methodology
The following methodology is used in this project:

**Step 1.** Information gathered from the Titanic dataset.
**Step 2.** Data cleansing and preprocessing.
**Step 3.** Addressing the Age and Embarked columns' missing values.
**Step 4.** Removing items like the cabin that aren't needed.
**Step 5.** Encoding categorical characteristics such as Embarked and Sex.
**Step 6.** Dividing data into datasets for testing and training.
**Step 7.** Developing a model for logistic regression.
**Step 8.** Using accuracy metrics to assess the model's performance.
**Step 9.** Using Streamlit to save and implement the trained model.
**Step 10**.Utilising user input to forecast passenger survival.


## implementation

**Software Requirements**
Python
VS Code
Streamlit
Pandas
NumPy
Scikit-learn

**Implementation Steps**
Load the Titanic dataset using Pandas.
Clean and preprocess the data.
Convert categorical values into numerical format.
Train the Logistic Regression model.
Save the trained model as model.pkl.
Develop a Streamlit user interface.
Accept passenger details as input.
Predict survival status and display results.

**Machine Learning Algorithm Used**
Logistic Regression
Dataset Features
Passenger Class (Pclass)
Sex
Age
Siblings/Spouses (SibSp)
Parents/Children (Parch)
Fare
Embarked

## Conclusion
The Titanic Survival Prediction project effectively illustrates how machine learning may be used in predictive analytics. The model can accurately predict passenger survival through the use of logistic regression and data preparation approaches. The project offers hands-on instruction in feature engineering, data analysis, model training, and deployment. It is a great illustration of a categorisation difficulty in the real world.

## Future Scope
**1.** Improve accuracy using advanced algorithms such as Random Forest and XGBoost.
**2.** Perform feature engineering for better predictions.
**3.** Integrate deep learning techniques.
**4.** Deploy the application on cloud platforms.
**5.** Add data visualization dashboards.
**6.** Develop a mobile-friendly web application.
**7.** Implement real-time prediction services using APIs.

## References
**1.** Kaggle Titanic Dataset
**2.** Scikit-learn Documentation
**3.** Pandas Documentation
**4.** Python Official Documentation
**5.** Streamlit Documentation
**6.** Machine Learning by Tom M. Mitchell
**7.** Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron
