Customer Churn Prediction – Machine Learning Project


Project Overview
In this project, I built a machine learning model to predict whether a telecom customer will churn or not. Customer churn is a serious issue for telecom companies because losing customers directly affects revenue.
The goal of this project is to identify customers who are likely to leave so that the company can take action in advance.

This is a binary classification problem where the output is:
Yes (Customer will churn)
No (Customer will not churn)


Dataset Information
I used a telecom dataset containing around 7,000 customer records.
The target variable is Churn Label.


The dataset includes different types of features:
Demographic details – Age, Gender, Senior Citizen, Dependents

Account information – Tenure, Contract Type
Billing information – Monthly Charges, Total Revenue, Extra Data Charges
Customer experience – Satisfaction Score


Exploratory Data Analysis (EDA)
Before building the model, I performed EDA to understand customer behavior.

Some important findings:
Customers with low tenure (new customers) are more likely to churn
Customers with higher monthly charges show higher churn rate
Customers with low satisfaction scores churn more frequently
These insights helped me understand which features might be important for prediction.

Statistical Testing
To support my observations with statistical proof, I performed an Independent T-Test on important numerical features.

For example:
Tenure showed a statistically significant difference between churned and non-churned customers (p-value < 0.05).
This step helped ensure that my feature selection was based on statistical evidence, not just visual assumptions.


Feature Engineering
I created a few additional features to improve prediction performance, such as:
Revenue-related combined metrics
Indicators for customer dependency
This helped the model better understand customer behavior patterns.


Machine Learning Pipeline
To make the solution clean and reusable, I created a complete pipeline using Scikit-learn.


Preprocessing Steps:
Scaled numerical features using StandardScaler
Encoded categorical features using OneHotEncoder
Used ColumnTransformer to apply transformations correctly
Combined everything inside a Pipeline to prevent data leakage
This ensures that the same preprocessing steps are applied during both training and prediction.
Model Selection


I used Logistic Regression as the primary model because:
It performs well for binary classification
It is easy to interpret
It gives probability outputs
It works well as a strong baseline model
Model Evaluation


I evaluated the model using:
Confusion Matrix
Precision
Recall
F1-Score

Since missing a churned customer is costly for business, I focused more on Recall for churned customers.
The model achieved strong recall, meaning it correctly identifies most customers who are likely to churn.


Tech Stack
Python
Pandas, NumPy
Scikit-learn
SciPy (Statistical Testing)
Matplotlib,Seaborn 


Key Learnings
End-to-end machine learning workflow
Importance of statistical validation during EDA
Clean preprocessing using pipelines
Building deployable and maintainable ML solutions


Model Saving
Finally, I saved the complete pipeline (preprocessing + model) using joblib so it can be reused later without retraining:
joblib.dump(model, "customer_churn_pipeline.pkl")
