import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


titanic = sns.load_dataset('titanic')

print("="*40 + "\nDataset Shape:", titanic.shape)
print("\nFirst 5 Rows:")
print(titanic.head().to_string())  # Force table display
print("\nData Types:\n", titanic.dtypes)
print("\nMissing Values:\n", titanic.isnull().sum())

# Data Cleaning

titanic['age'] = titanic.groupby(['sex', 'pclass'])['age'].transform(
    lambda x: x.fillna(x.median())
)
titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])

titanic.drop(['deck', 'alive', 'alone'], axis=1, inplace=True)

print("\n" + "="*40 + "\nAfter Cleaning:")
print("Missing Values:\n", titanic.isnull().sum())


# Survival by Passenger Class
plt.figure(figsize=(10, 5))
sns.barplot(x='pclass', y='survived', data=titanic, ci=None, palette='viridis')
plt.title('Survival Rate by Passenger Class (1 = Highest Class)')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

# Survival by Gender and Age
titanic['is_child'] = np.where(titanic['age'] < 18, 1, 0)  

plt.figure(figsize=(10, 6))
sns.barplot(x='sex', y='survived', hue='is_child', data=titanic, ci=None, palette='mako')
plt.title('Survival Rate by Gender and Age Group')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.legend(title='Child (0=Adult, 1=Child)')
plt.show()

# 2. Detect Anomalies

plt.figure(figsize=(10, 4))
sns.boxplot(x=titanic['fare'], palette='YlOrRd')
plt.title('Fare Distribution Analysis')
plt.show()

print("\nTop 5 Highest Fares:")
print(titanic.nlargest(5, 'fare')[['fare', 'pclass', 'sex', 'age', 'embarked']])

# Hypothesis 1
cont_table = pd.crosstab(titanic['pclass'], titanic['survived'])
chi2, p, dof, expected = stats.chi2_contingency(cont_table)
print(f"\nHypothesis 1 (Class vs Survival): p-value = {p:.4f}")

# Hypothesis 2
children = titanic[titanic['is_child'] == 1]['survived']
adults = titanic[titanic['is_child'] == 0]['survived']
t_stat, p_val = stats.ttest_ind(children, adults)
print(f"Hypothesis 2 (Age Impact): p-value = {p_val:.4f}")

# Age Distribution by Survival
plt.figure(figsize=(10, 6))
sns.histplot(data=titanic, x='age', hue='survived', element='step', kde=True, palette='Set1')
plt.title('Age Distribution by Survival Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Family Size Impact
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1
plt.figure(figsize=(10, 6))
sns.countplot(x='family_size', hue='survived', data=titanic, palette='rocket')
plt.title('Survival Count by Family Size')
plt.xlabel('Family Size (Including Self)')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(10, 6))
corr_matrix = titanic[['survived', 'pclass', 'age', 'fare', 'family_size']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()


#Survival rate by class
class_survival = titanic.groupby('pclass')['survived'].mean().reset_index()
class_1_rate = class_survival[class_survival['pclass'] == 1]['survived'].values[0] * 100
class_3_rate = class_survival[class_survival['pclass'] == 3]['survived'].values[0] * 100

#Survival rate by gender
gender_survival = titanic.groupby('sex')['survived'].mean().reset_index()
female_rate = gender_survival[gender_survival['sex'] == 'female']['survived'].values[0] * 100
male_rate = gender_survival[gender_survival['sex'] == 'male']['survived'].values[0] * 100

#Survival rate by age group
age_survival = titanic.groupby('is_child')['survived'].mean().reset_index()
child_rate = age_survival[age_survival['is_child'] == 1]['survived'].values[0] * 100
adult_rate = age_survival[age_survival['is_child'] == 0]['survived'].values[0] * 100

pclass_corr = titanic['pclass'].corr(titanic['survived'])

print("\n" + "="*40 + "\nKey Insights (Calculated from Data):")
print(f"- 1st Class passengers: {class_1_rate:.1f}% survival vs 3rd Class: {class_3_rate:.1f}%")
print(f"- Female survival rate: {female_rate:.1f}% vs Male: {male_rate:.1f}%")
print(f"- Children (<18): {child_rate:.1f}% survival vs Adults: {adult_rate:.1f}% (p < 0.0001)")
print(f"- Correlation between class and survival: {pclass_corr:.2f} (Negative = Lower class numbers (1st) correlate with survival)")