import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# Display the updated dataframe
print("\nData after Preprocessing:")
print(data.head())

# Calculate average job satisfaction level of employees who left
avg_satisfaction_left = data[data['left'] == 1]['job_satisfaction_level'].mean()
print(f'\nAverage job satisfaction level of employees who left: {avg_satisfaction_left:.2f}')

# Calculate average salary satisfaction level of employees who left
avg_salary_left = data[data['left'] == 1]['salary'].map({1: 'low', 2: 'medium', 3: 'high'}).mode()[0]
print(f'Average salary satisfaction level of employees who left: {avg_salary_left}')

# Calculate average duration spent with the company for employees who left
avg_years_left = data[data['left'] == 1]['years_spent_with_company'].mean()
print(f'Average years spent with FAU Bank for employees who left: {avg_years_left:.2f}')

# Analyze effect of salary on employees deciding to quit
salary_turnover_effect = data.groupby('salary')['left'].mean()
# Reverse the salary mapping for readability
salary_turnover_effect.index = salary_turnover_effect.index.map({v: k for k, v in salary_mapping.items()})

print(f'\nSalary effect on turnover:\n{salary_turnover_effect}')

# Calculate correlation matrix
corr_matrix = data.corr()

# Display correlation matrix
print('\nCorrelation matrix:')
print(corr_matrix)

# Display correlations of the 'left' column with other features
print('\nCorrelations with the "left" column:')
print(corr_matrix['left'].sort_values(ascending=False))

# Create a heatmap of the correlation matrix with formal colors
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f")
plt.title('Correlation Matrix Heatmap', fontsize=16)
plt.show()
