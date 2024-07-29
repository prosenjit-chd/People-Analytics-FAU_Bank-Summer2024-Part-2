import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
dataset_path = 'FAU_Bank_Employee_Wellbeing.csv'  # Update with actual path
dataframe = pd.read_csv(dataset_path)

# Check for missing values
null_counts = dataframe.isnull().sum()
print("Missing values:\n", null_counts)

# Drop irrelevant columns
dataframe.drop(columns=['Employee_ID'], inplace=True)

# Convert categorical data to numerical
# Age conversion
age_conversion = {'Less than 20': 1, '21 to 35': 2, '36 to 50': 3, '51 or more': 4}
dataframe['AGE'] = dataframe['AGE'].map(age_conversion)

# Gender conversion
gender_conversion = {'Male': 0, 'Female': 1}
dataframe['GENDER'] = dataframe['GENDER'].map(gender_conversion)

# Plot for stress by gender
plt.figure(figsize=(10, 6))
sns.barplot(x='GENDER', y='DAILY_STRESS', data=dataframe, ci=None, palette='muted')
plt.xlabel('Gender')
plt.ylabel('Daily Stress')
plt.title('Daily Stress by Gender')
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])
plt.show()

# Plot for stress by job role
plt.figure(figsize=(14, 8))
sns.barplot(x='JOB_ROLE', y='DAILY_STRESS', data=dataframe, ci=None, palette='deep')
plt.xlabel('Job Role')
plt.ylabel('Daily Stress')
plt.title('Daily Stress by Job Role')
plt.xticks(rotation=45)
plt.show()

# Average hobby time by gender
avg_hobby_time_by_gender = dataframe.groupby('GENDER')['TIME_FOR_HOBBY'].mean()
# print("Average hobby time by gender:\n", avg_hobby_time_by_gender)

# Bar plot for hobby time by gender
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=avg_hobby_time_by_gender.index, y=avg_hobby_time_by_gender.values, palette='pastel')
plt.xlabel('Gender')
plt.ylabel('Average Time for Hobby')
plt.title('Average Time for Hobby by Gender')
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])

# Adding labels on bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')

plt.show()

# Heatmap for correlation with WLB score
# Job role conversion
job_role_conversion = {
    'Bank Teller': 1, 'Business Analyst': 2, 'Credit Analyst': 3, 'Customer Service': 4,
    'Finance Analyst': 5, 'Human Resources': 6, 'Investment Banker': 7, 'Loan Processor': 8,
    'Mortgage Consultant': 9, 'Risk Analyst': 10
}
dataframe['JOB_ROLE'] = dataframe['JOB_ROLE'].map(job_role_conversion)

# Correlation matrix
corr_matrix = dataframe.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Blues')
plt.title('Correlation Matrix')
plt.show()

# Correlation with WLB score
wlb_corr = corr_matrix['WORK_LIFE_BALANCE_SCORE'].sort_values(ascending=False)
print("Attributes highly correlated with WLB score:\n", wlb_corr)
