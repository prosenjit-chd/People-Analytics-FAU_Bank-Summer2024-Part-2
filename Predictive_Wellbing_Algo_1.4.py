import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

# Load dataset
dataset_path = 'FAU_Bank_Employee_Wellbeing.csv'
data = pd.read_csv(dataset_path)

# Convert categorical data to numerical
age_dict = {'Less than 20': 1, '21 to 35': 2, '36 to 50': 3, '51 or more': 4}
data['AGE'] = data['AGE'].map(age_dict)
gender_dict = {'Male': 0, 'Female': 1}
data['GENDER'] = data['GENDER'].map(gender_dict)
job_role_dict = {
    'Bank Teller': 1, 'Business Analyst': 2, 'Credit Analyst': 3, 'Customer Service': 4, 
    'Finance Analyst': 5, 'Human Resources': 6, 'Investment Banker': 7, 'Loan Processor': 8, 
    'Mortgage Consultant': 9, 'Risk Analyst': 10
}
data['JOB_ROLE'] = data['JOB_ROLE'].map(job_role_dict)

# Handle missing values
features = data.drop(columns=['WORK_LIFE_BALANCE_SCORE'])
target = data['WORK_LIFE_BALANCE_SCORE']
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_imputed, target, test_size=0.2, random_state=42)

# Train the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predict on the test set
predictions = lin_reg.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, predictions)
print(f"R² score: {r2}")

# Present the difference between actual and predicted values in a tabular form
results_df = pd.DataFrame({
    'Actual': y_test.reset_index(drop=True),
    'Predicted': predictions,
    'Difference': y_test.reset_index(drop=True) - predictions
})
print("\nDifference between actual and predicted values:\n", results_df)

# Create a scatter plot of actual vs predicted values with formal colors
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictions, color='navy', s=50)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot of Actual vs. Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='darkred', linestyle='--', linewidth=2)  # Add a reference line
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Predict the WLB score for a new employee
new_emp = {
    'Employee_ID': 2222,
    'JOB_ROLE': 1,  # Bank Teller
    'DAILY_STRESS': 2,
    'WORK_TRAVELS': 2,
    'TEAM_SIZE': 5,
    'DAYS_ABSENT': 0,
    'WEEKLY_EXTRA_HOURS': 5,
    'ACHIEVED_BIG_PROJECTS': 2,
    'EXTRA_HOLIDAYS': 0,
    'BMI_RANGE': 1,
    'TODO_COMPLETED': 6,
    'DAILY_STEPS_IN_THOUSAND': 5,
    'SLEEP_HOURS': 7,
    'LOST_VACATION': 5,
    'SUFFICIENT_INCOME': 1,
    'PERSONAL_AWARDS': 4,
    'TIME_FOR_HOBBY': 0,
    'AGE': 2,  # 21 to 35
    'GENDER': 0  # Male
}

# Create DataFrame for the new employee
new_emp_df = pd.DataFrame([new_emp])

# Impute missing values (in this case, none
