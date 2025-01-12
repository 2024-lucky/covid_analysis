# New Coronary Patient Comprehensive Statistical Analysis

# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, f_oneway, kruskal
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Dataset
def load_data(filepath):
    return pd.read_csv(filepath)

data = load_data("filtered_data.csv")

# Step 2: Data Preprocessing
# Encode gender as numeric and normalize age
scaler = StandardScaler()
data['gender_numeric'] = data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
data['age_category_normalized'] = scaler.fit_transform(data[['age_category']])

# Encode condition as categorical labels for multi-class analysis
label_encoder = LabelEncoder()
data['condition_encoded'] = label_encoder.fit_transform(data['condition'])

# Step 3: Distribution Analysis
# Gender Distribution
def plot_gender_distribution(data):
    gender_counts = data['gender'].value_counts()
    gender_counts.plot(kind='bar', color=['blue', 'orange'], alpha=0.7)
    plt.title('Gender Distribution', fontsize=16)
    plt.xlabel('Gender', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

plot_gender_distribution(data)

# Age Distribution
def plot_age_distribution(data):
    plt.hist(data['age_category'], bins=20, color='green', alpha=0.7)
    plt.title('Age Distribution', fontsize=16)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.tight_layout()
    plt.show()

plot_age_distribution(data)

# Step 4: Chi-Square Test for Gender and Condition Relationship
def chi_square_test(data):
    contingency_table = pd.crosstab(data['gender'], data['condition'])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    print("Chi-Square Test Results:")
    print(f"Chi-Square Statistic: {chi2_stat}")
    print(f"p-value: {p_value}")
    print(f"Degrees of Freedom: {dof}")

chi_square_test(data)

# Step 5: ANOVA and Kruskal-Wallis Test for Age and Condition Relationship
def anova_and_kruskal(data):
    groups = [data[data['condition'] == condition]['age_category'] for condition in data['condition'].unique()]
    anova_stat, anova_p_value = f_oneway(*groups)
    kruskal_stat, kruskal_p_value = kruskal(*groups)
    print("ANOVA Test Results:")
    print(f"F-Statistic: {anova_stat}, p-value: {anova_p_value}")
    print("Kruskal-Wallis Test Results:")
    print(f"H-Statistic: {kruskal_stat}, p-value: {kruskal_p_value}")

anova_and_kruskal(data)

# Step 6: Logistic Regression (Binary Classification)
def logistic_regression_binary(data, target_column):
    X = data[['gender_numeric', 'age_category_normalized']]
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # ROC Curve
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('ROC Curve', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

logistic_regression_binary(data, 'binary_asymptomatic')

# Step 7: Multi-Class Logistic Regression
def logistic_regression_multiclass(data):
    X = data[['gender_numeric', 'age_category_normalized']]
    y = data['condition_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

logistic_regression_multiclass(data)
