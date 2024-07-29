import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('FAU_Bank_turnover.csv')

# Define mappings for job roles and salary levels
role_mapping = {
    'bank_teller': 1, 'business_analyst': 2, 'credit_analyst': 3, 'customer_service': 4,
    'finance_analyst': 5, 'hr': 6, 'investment_banker': 7, 'IT': 8, 'loan_analyst': 9, 'mortgage_consultant': 10
}
salary_mapping = {'low': 1, 'medium': 2, 'high': 3}

# Apply the mappings to the data
data['job_role'] = data['job_role'].map(role_mapping)
data['salary'] = data['salary'].map(salary_mapping)

# Bin job satisfaction and performance evaluation into categories
data['job_satisfaction_bin'] = pd.cut(data['job_satisfaction_level'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5])
data['performance_evaluation_bin'] = pd.cut(data['last_performance_evaluation'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5])

# Create a combined feature of number of projects and working hours
data['projects_hours'] = data['completed_projects'] * data['average_working_hours_monthly']

# Prepare feature matrix and target vector
features = data[['job_satisfaction_level', 'engagement_with_task', 'last_performance_evaluation',
                 'completed_projects', 'average_working_hours_monthly', 'years_spent_with_company',
                 'received_support', 'promotion_last_5years', 'job_role', 'salary', 'projects_hours']]

target = data['left']

# Handle missing values
imputer = SimpleImputer(strategy='mean')  # You can change strategy as needed
features_imputed = imputer.fit_transform(features)

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

# Initialize and train the Logistic Regression model
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

# Make predictions
predictions = logistic_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

# Print the performance metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# Plot the confusion matrix with formal colors
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.show()

# Feature importance using the coefficients from Logistic Regression
feature_importance = pd.DataFrame({'Feature': features.columns, 'Importance': logistic_model.coef_[0]})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Plot feature importance with formal colors
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='Blues_d')
plt.title('Feature Importance', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.show()
