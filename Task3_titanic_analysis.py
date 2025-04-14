import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "titanic_results")
os.makedirs(OUTPUT_DIR, exist_ok=True) 

plt.switch_backend('agg')

# Load and clean data
df = sns.load_dataset('titanic')
df['age'] = df['age'].fillna(df['age'].median())
df = df.dropna(subset=['embarked'])

def save_plot(name):
    """Save plot with verified path"""
    path = os.path.join(OUTPUT_DIR, name)
    try:
        plt.savefig(path, bbox_inches='tight')
        print(f"✅ Saved: {os.path.abspath(path)}")
    except Exception as e:
        print(f"❌ Failed to save {path}: {str(e)}")
    plt.close()

# 1. Survival by Class/Gender
plt.figure(figsize=(10,6))
sns.barplot(x='pclass', y='survived', hue='sex', data=df, errorbar=None)
plt.title("Survival by Class & Gender")
save_plot('survival_class_gender.png')

# 2. Age Distribution
plt.figure(figsize=(10,6))
sns.histplot(df, x='age', hue='survived', bins=20, kde=True)
plt.title("Age Distribution by Survival")
save_plot('age_distribution.png')

# 3. Fare Analysis
plt.figure(figsize=(10,6))
sns.boxplot(x='survived', y='fare', data=df)
plt.xticks([0,1], ['Died', 'Survived'])
plt.yscale('log')
plt.title("Fare Distribution by Survival")
save_plot('fare_analysis.png')

print(f"\nCheck this folder for results:\n{os.path.abspath(OUTPUT_DIR)}")