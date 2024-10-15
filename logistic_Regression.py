import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('/content/heart.csv')

print(dataset.head())

print("Missing values per column:")
print(dataset.isnull().sum())

data = pd.get_dummies(dataset, columns=['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall'])

X = data.drop('output', axis=1)
y = data['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)
print("Model Accuracy: %.2f%%" % (accuracy * 100))

search_type = input("Do you want to search for a specific age or an age range? (Enter 'specific' or 'range'): ").strip().lower()

if search_type == 'specific':
    user_age = int(input("Enter the age to filter by: "))
    filtered_data = dataset[dataset['age'] == user_age]

elif search_type == 'range':
    age_min = int(input("Enter the minimum age(Min_Range = 41): "))
    age_max = int(input("Enter the maximum age(Max_Range = 98): "))
    filtered_data = dataset[(dataset['age'] >= age_min) & (dataset['age'] <= age_max)]

else:
    print("Invalid input. Please enter 'specific' or 'range'.")
    filtered_data = pd.DataFrame()

if filtered_data.empty:
    print(f"No records found for the given age or age range.")
else:
    total_records = len(filtered_data)
    male_count = filtered_data[filtered_data['sex'] == 1]
    female_count = filtered_data[filtered_data['sex'] == 0]

    print(f"\nTotal records for the selected age or age range: {total_records}")

    print(f"\nMales:")
    print(f"Total Males: {len(male_count)}")
    print(f" - With Heart Disease: {len(male_count[male_count['output'] == 1])}")
    print(f" - Without Heart Disease: {len(male_count[male_count['output'] == 0])}")

    print(f"\nFemales:")
    print(f"Total Females: {len(female_count)}")
    print(f" - With Heart Disease: {len(female_count[female_count['output'] == 1])}")
    print(f" - Without Heart Disease: {len(female_count[female_count['output'] == 0])}")
