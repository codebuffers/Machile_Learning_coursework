# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:58:14 2023

@author: Iskooo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
sns.set()


# INITIAL PROCEDURES
orig_data = pd.read_csv("loan_approval_dataset.csv")
orig_data.head()

orig_data.columns.values

# Removing whitespace at the beginning of the columns
orig_data.columns = orig_data.columns.str.replace(" ","")

# Number of null values:
orig_data.isnull().sum()

# Number of duplicated rows:
orig_data[orig_data["loan_id"].duplicated(keep = False) == True].sort_values(["loan_id"])


# Data type of each column
orig_data.info()

orig_data.describe(include = "all")

# number of raws and columns
orig_data.shape

orig_data.columns.values


# Dropping first column as unnecessary
df = orig_data.drop(["loan_id"], axis = 1)

df.head()


# EXPLORATORY DATA ANALYSIS

sns.pairplot(df)

#Loan amount vs status:
sns.histplot(x = "loan_amount", hue = "loan_status", data = df)
plt.xlabel("Loan amount")
plt.title("Loan amount vs loan status:")
plt.show()


# Both approved and rejected loans have similar trends.

# The below plots are made after referring to the above pairplot.

# Income amount vs loan amount and status:
sns.scatterplot(y = "loan_amount", x = "income_annum", hue = "loan_status", data = df)
plt.title("Annual income vs loan amount and loan status:")
plt.ylabel("Loan amount")
plt.xlabel("Annual income")
plt.show()


# When annual income increases, loan amount tends to increase.
# Low income tends to outcome in low loan amount range.
# High income in high loan income range.


# Highest income and loan approved:
df.loc[(df["income_annum"] == df["income_annum"].max()) & (df["loan_status"] == " Approved")]




# Credit score
sns.scatterplot(y = "loan_amount", x = "cibil_score", hue = "loan_status", data = df)
plt.title("Credit score vs loan amount and loan status:")
plt.ylabel("Loan amount")
plt.xlabel("Credit score")
plt.show()


# Most loans with poor credit score less than 600 were rejected.
# A few clients with high credit score also had loans rejected.
# Loan amount and credit score are highly related.


# Excellent credit score and still rejected:
df.loc[(df["cibil_score"] > 740) & (df["loan_status"] == " Rejected")].sort_values(["cibil_score"])




# Asset values
fig,axes = plt.subplots(2, 2, figsize = (10, 10))
plt.title("Residential, luxury, commercial and bank asset values histograms:")
sns.histplot(df, x = "residential_assets_value", hue = "loan_status", ax = axes[0,0])
axes[0,0].set_xlabel("Residential Assets Value")
axes[0,0].set_ylabel("Count")

sns.histplot(df, x = "luxury_assets_value", hue = "loan_status", ax = axes[1,0])
axes[1,0].set_xlabel("Luxury Assets Value")
axes[1,0].set_ylabel("Count")

sns.histplot(df, x = "commercial_assets_value", hue = "loan_status", ax = axes[0,1])
axes[0,1].set_xlabel("Commercial Assets Value")
axes[0,1].set_ylabel("Count")

sns.histplot(df, x = "bank_asset_value", hue = "loan_status", ax = axes[1,1])
axes[1,1].set_xlabel("Bank Assets Value")
axes[1,1].set_ylabel("Count")

plt.tight_layout()
plt.show()

# For all 4 assets values, "Approved" and "Rejected" show similar trends.

# Checking correlation:  
sns.heatmap(df.corr(), annot = True, fmt = ".2f", cmap = "Reds")

# Luxury and bank assets value have strong correlation with annual income.
# Residential and commercial assets value have a medium correlation with annual income.
# Loan term doesn't show any correlation.



# Loan term vs loan status

# Loan term vs status crosstab can be made since status has categorical values.

# Columns have categorical value:
loan_tab = pd.crosstab(columns = df["loan_status"], index = df["loan_term"])
loan_tab["Total"] = loan_tab[" Approved"] + loan_tab[" Rejected"]
loan_tab["Approved%"] = (loan_tab[" Approved"] / loan_tab["Total"])*100
loan_tab["Rejected%"] = 100 - loan_tab["Approved%"]
loan_tab

plt.figure(figsize = (12, 6))
loan_tab.plot.line(marker = "H")
plt.title("Loan status vs term: ")
plt.ylabel("Count")
plt.show()

# Loan amount vs loan term with status:
sns.scatterplot(x = "loan_amount", y = "loan_term", hue = "loan_status", data = df)
plt.title("Loan amount vs term with status:")
plt.xlabel("Loan amount")
plt.ylabel("Loan term")
plt.show()
  
# 4 year loan term gets most chance for approval.
# Most loans have a 6 year term.
# For 10 year term, chance of approval and rejection is almost same.



# Education countplot:
sns.countplot(data = df, x = "education", hue = "loan_status")
plt.title("Education:")
plt.xlabel("Education")
plt.show()
education_df = df.groupby(["education"], as_index=False).agg(
    count=("education", "count"),
    median_annual_income=("income_annum", "median"),
    meam_loan_amount=("loan_amount", "mean"),
    mean_credit_score = ("cibil_score", "mean"),
    mean_loan_term = ("loan_term", "mean"),
    mean_residential_value = ("residential_assets_value", "mean"),
    mean_commerical_value = ("commercial_assets_value", "mean"),
    mean_luxury_value = ("luxury_assets_value", "mean"),
    mean_bank_value = ("bank_asset_value", "mean")
    ).round(3).reset_index(drop=True)

education_df


# Employment countplot:
selfemp_df = df.groupby(["self_employed"], as_index=False).agg(
    count=("education", "count"),
    median_annual_income=("income_annum", "median"),
    meam_loan_amount=("loan_amount", "mean"),
    mean_credit_score = ("cibil_score", "mean"),
    mean_loan_term = ("loan_term", "mean"),
    mean_residential_value = ("residential_assets_value", "mean"),
    mean_commerical_value = ("commercial_assets_value", "mean"),
    mean_luxury_value = ("luxury_assets_value", "mean"),
    mean_bank_value = ("bank_asset_value", "mean")
    ).round(3).reset_index(drop=True)

selfemp_df

# education and employment shows no actual relation with other variables.



# DATA PREPARATION


df.head()

# Converting categorical data with dummies:
dummy = pd.get_dummies(df)
dummy.head()

dummy = dummy.drop(["education_ Not Graduate", "self_employed_ No", "loan_status_ Rejected"], axis = 1)
dummy.rename(columns = {"education_ Graduate" : "education", "self_employed_ Yes" : "self_employed", "loan_status_ Approved" : "loan_status"}, inplace = True)
    

dummy.head()

plt.figure(figsize=(8,8))
sns.heatmap(dummy.corr(), annot=True, fmt=".2f", cmap = "Blues")

# "cibil_score" affects loan_status the most.
# "education" and "self_employed" has no linear relationship with any variable.

dummy.columns.values



# PREPARING TRAINING AND TESTING DATA:
y = dummy["loan_status"]
X = dummy.drop(["loan_status"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# Scaling:
scaler = StandardScaler()
scaled_data = scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)



# LOGISTIC REGRESSION MODEL
reg = LogisticRegression()
reg.fit(X_train_scaled, y_train)

print("Training Accuracy = ", reg.score(X_train_scaled, y_train)*100, "%")


# Testing and results:
print("Testing Accuracy = ", reg.score(X_test_scaled, y_test)*100,"%")

y_predicted = reg.predict(X_test_scaled)

print("Accuracy = ", accuracy_score(y_predicted, y_test)*100, "%")
print("F1 score = ", f1_score(y_predicted, y_test)*100, "%")
print("Recall = ", recall_score(y_predicted, y_test)*100, "%")
print("Precision = ", precision_score(y_predicted, y_test)*100, "%")


# Confusion matrix:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_predicted, y_test)
plt.figure(figsize=(2,2))
sns.heatmap(cm, annot=True, fmt=".2f", cmap = "Spectral")


# Cross validation

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

#Performing k-fold cross-validation:
k = 5
scores = cross_val_score(reg, X_train_scaled, y_train, cv=k, scoring="accuracy")

#Printing the cross-validation scores:
print("Cross-validation scores:", scores)

#Calculating and printing the mean accuracy and its SD:
mean_accuracy = np.mean(scores)
std_accuracy = np.std(scores)
print(f"Mean Accuracy: {mean_accuracy:.2f}")
print(f"Standard Deviation: {std_accuracy:.2f}")



# SUPPORT VECTOR MACHINE (SVM) MODEL

from sklearn.svm import SVC

#Creating an SVM classifier with a linear kernel:
svm_classifier = SVC(kernel="linear", C=1.0, random_state=42)

# Fit the classifier to the training data
svm_classifier.fit(X_train_scaled, y_train)
y_pred = svm_classifier.predict(X_test_scaled)



# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_pred, y_test)
plt.figure(figsize=(2,2))
sns.heatmap(cm1, annot=True, fmt=".2f", cmap = "Blues")




# RANDOM FOREST MODEL

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

y_pred_new = rf_classifier.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm2 = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

plt.figure(figsize=(2,2))
sns.heatmap(cm2, annot=True, fmt=".2f", cmap = "coolwarm")


from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_


best_params, best_rf_model

best_rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

best_rf_model.fit(X_train_scaled, y_train)

new_data_predictions = best_rf_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, new_data_predictions)
precision = precision_score(y_test, new_data_predictions)
recall = recall_score(y_test, new_data_predictions)
f1 = f1_score(y_test, new_data_predictions)
cm3 = confusion_matrix(y_test, new_data_predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

plt.figure(figsize=(2,2))
sns.heatmap(cm3, annot=True, fmt=".2f", cmap = "coolwarm")
