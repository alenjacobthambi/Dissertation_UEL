import math, time, random, datetime

# Data Analysis Packages and visualization Packages
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.tools as tls
from plotly.subplots import make_subplots
get_ipython().run_line_magic('matplotlib', 'inline')

# Packages for preprocessing
from sklearn.preprocessing import OneHotEncoder,  label_binarize, LabelEncoder, StandardScaler

# Machine Learning packages
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.model_selection import train_test_split,StratifiedKFold, GridSearchCV, learning_curve, cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import Perceptron,SGDClassifier,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Boosting Algorithms
from xgboost import XGBClassifier
import lightgbm as lgb

# Package for ignoring minor Warnings
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Employee_Attrition.csv")

df.info()

# Dropping few unnecessary columns
df.drop(['EmployeeNumber','Over18','StandardHours','EmployeeCount'],axis=1,inplace=True)

df.shape

# Package for complete report of each features and correlations
ProfileReport(df)


# # Findings from Profile Report
# - Job level is strongly correlated with total working yyears
# - Monthly income is strongly correlated with Job level
# - Monthly income is strongly correlated with total working years
# - Age is stongly correlated with monthly income
# - Performance Rating is strongly correlated with Salary Hike
# - Department is strongly correlated with Job Role

# Converting the variable into boolean
df['Attrition'] = df['Attrition'].apply(lambda x:1 if x == "Yes" else 0 )
df['OverTime'] = df['OverTime'].apply(lambda x:1 if x =="Yes" else 0 )
attrition = df[df['Attrition'] == 1]
no_attrition = df[df['Attrition']==0]

df

# ### Visualization of Categorical Features 


def categorical_variable_viz(col_name):
    f, ax = plt.subplots(1, 2, figsize=(10, 6))

    # Count plot by factors
    df[col_name].value_counts().plot.bar(cmap='Set2', ax=ax[0])
    ax[0].set_title(f'Number of Employee by {col_name}')
    ax[0].set_ylabel('Count')
    ax[0].set_xlabel(f'{col_name}')

    # Count plot by factors by Attrition
    sns.countplot(x=col_name, hue='Attrition', data=df, ax=ax[1], palette='Set2')
    ax[1].set_title(f'Attrition by {col_name}')
    ax[1].set_xlabel(f'{col_name}')
    ax[1].set_ylabel('Count')
    plt.show()


categorical_variable_viz('BusinessTravel')
categorical_variable_viz('Department')
categorical_variable_viz('EducationField')
categorical_variable_viz('Education')
categorical_variable_viz('EnvironmentSatisfaction')
categorical_variable_viz('Gender')
categorical_variable_viz('JobRole')
categorical_variable_viz('JobInvolvement')
categorical_variable_viz('MaritalStatus')
categorical_variable_viz('NumCompaniesWorked')
categorical_variable_viz('OverTime')
categorical_variable_viz('StockOptionLevel')
categorical_variable_viz('TrainingTimesLastYear')
categorical_variable_viz('YearsWithCurrManager')

# ### Visualization of Numerical Features 

# Visualizing Numerical Features by Attrition
def numerical_variable_viz(col_name):
    f,ax = plt.subplots(1,2, figsize=(18,6))
    sns.kdeplot(attrition[col_name], label='Employee who left',ax=ax[0], fill=True, color='palegreen')
    sns.kdeplot(no_attrition[col_name], label='Employee who stayed', ax=ax[0], fill=True, color='salmon')
    sns.boxplot(y=col_name, x='Attrition',data=df, palette='Set3', ax=ax[1])

numerical_variable_viz("Age")
numerical_variable_viz("DailyRate")
numerical_variable_viz("DistanceFromHome")
numerical_variable_viz("MonthlyIncome")
numerical_variable_viz("HourlyRate")
numerical_variable_viz("JobInvolvement")
numerical_variable_viz("PercentSalaryHike")
numerical_variable_viz("TotalWorkingYears")
numerical_variable_viz("YearsAtCompany")
numerical_variable_viz("YearsInCurrentRole")
numerical_variable_viz("YearsSinceLastPromotion")
numerical_variable_viz("YearsWithCurrManager")


# ### Visualization of Categorical vs Numericals Features 

# Visualizing Numerical vs two Categorical Variables
def categ_numerical(numerical_col, categorical_col1, categorical_col2):
    f, ax = plt.subplots(1, 2, figsize=(20,8))
    g1 = sns.swarmplot(x=categorical_col1, y=numerical_col, hue='Attrition', dodge=True, ax=ax[0], palette='Set2', data=df)
    ax[0].set_title(f'{numerical_col} vs {categorical_col1} separated by Attrition')
    g1.set_xticklabels(g1.get_xticklabels(), rotation=90)
    g2 = sns.swarmplot(x=categorical_col2, y=numerical_col, hue='Attrition', dodge=True, ax=ax[1], palette='Set2', data=df)
    ax[1].set_title(f'{numerical_col} vs {categorical_col2} separated by Attrition')
    g2.set_xticklabels(g2.get_xticklabels(), rotation=90)

categ_numerical('Age','Gender','MaritalStatus')
categ_numerical('Age','JobRole','EducationField')
categ_numerical('MonthlyIncome','Gender','MaritalStatus')

# ## Feature Engineering Steps

# Joining Satisfaction Variables into one by average calculation

df['Total_Satisfaction'] = (df['EnvironmentSatisfaction'] + 
                            df['JobInvolvement'] + 
                            df['JobSatisfaction'] + 
                            df['RelationshipSatisfaction'] +
                            df['WorkLifeBalance']) /5 

# Drop the variables after joining
df.drop(['EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance'], axis=1, inplace=True)

categorical_variable_viz('Total_Satisfaction')
df.Total_Satisfaction.describe()

# Change Total satisfaction into boolean
# median = 2.8
# x = 1 if x >= 2.8

df['Total_Satisfaction_bool'] = df['Total_Satisfaction'].apply(lambda x:1 if x>=2.8 else 0 ) 
df.drop('Total_Satisfaction', axis=1, inplace=True)

# We can see that the count of attrition below 35 years old is high

df['Age_bool'] = df['Age'].apply(lambda x:1 if x<35 else 0)
df.drop('Age', axis=1, inplace=True)

# We can see that the count of attrition of employees in dailtRate less than 800 is high

df['DailyRate_bool'] = df['DailyRate'].apply(lambda x:1 if x<800 else 0)
df.drop('DailyRate', axis=1, inplace=True)

# R&D Department have higher attrition rate that other departments

df['Department_bool'] = df['Department'].apply(lambda x:1 if x=='Research & Development' else 0)
df.drop('Department', axis=1, inplace=True)

# DistanceFromHome > 10 have higher attrition rate

df['DistanceFromHome_bool'] = df['DistanceFromHome'].apply(lambda x:1 if x>10 else 0)
df.drop('DistanceFromHome', axis=1, inplace=True)

# Employees are more likey to drop the job if the employee is working as Laboratory Technician

df['JobRole_bool'] = df['JobRole'].apply(lambda x:1 if x=='Laboratory Technician' else 0)
df.drop('JobRole', axis=1, inplace=True)

# hourly rate < 65 have higher attrition rate

df['HourlyRate_bool'] = df['HourlyRate'].apply(lambda x:1 if x<65 else 0)
df.drop('HourlyRate', axis=1, inplace=True)

# MonthlyIncome < 4000 have higher attrition rate

df['MonthlyIncome_bool'] = df['MonthlyIncome'].apply(lambda x:1 if x<4000 else 0)
df.drop('MonthlyIncome', axis=1, inplace=True)

# NumCompaniesWorked < 3 have higher attrition rate

df['NumCompaniesWorked_bool'] = df['NumCompaniesWorked'].apply(lambda x:1 if x>3 else 0)
df.drop('NumCompaniesWorked', axis=1, inplace=True)

# TotalWorkingYears < 8 have higher attrition rate

df['TotalWorkingYears_bool'] = df['TotalWorkingYears'].apply(lambda x:1 if x<8 else 0)
df.drop('TotalWorkingYears', axis=1, inplace=True)

# YearsAtCompany < 3 have higher attrition rate

df['YearsAtCompany_bool'] = df['YearsAtCompany'].apply(lambda x:1 if x<3 else 0)
df.drop('YearsAtCompany', axis=1, inplace=True)

# YearsInCurrentRole < 3 have higher attrition rate

df['YearsInCurrentRole_bool'] = df['YearsInCurrentRole'].apply(lambda x:1 if x<3 else 0)
df.drop('YearsInCurrentRole', axis=1, inplace=True)

# YearsSinceLastPromotion < 1 have higher attrition rate

df['YearsSinceLastPromotion_bool'] = df['YearsSinceLastPromotion'].apply(lambda x:1 if x<1 else 0)
df.drop('YearsSinceLastPromotion', axis=1, inplace=True)

# YearsWithCurrManager < 1 have higher attrition rate

df['YearsWithCurrManager_bool'] = df['YearsWithCurrManager'].apply(lambda x:1 if x<1 else 0)
df.drop('YearsWithCurrManager', axis=1, inplace=True)

# Converting Gender to boolean
df['Gender'] = df['Gender'].apply(lambda x:1 if x=='Female' else 0)

df.drop('MonthlyRate', axis=1, inplace=True)
df.drop('PercentSalaryHike', axis=1, inplace=True)

categorical_variables = ['BusinessTravel','Education','EducationField','MaritalStatus','StockOptionLevel','OverTime','Gender','TrainingTimesLastYear']
for col in categorical_variables:
        df[col] = df[col].astype('category')

df.info()

# We are separating the categorical and numerical data
categorical_vars = df.select_dtypes(include=['category'])
numerical_vars = df.select_dtypes(include=['int64'])
other_vars = df.select_dtypes(include=['float64'])

# Drop the attrition variable as that is dependant variable
numerical_vars.drop('Attrition', axis=1, inplace=True)

y = df['Attrition']

# Doing factozation for the categorical variables
categorical_vars = pd.get_dummies(categorical_vars)

numerical_vars.info()

categorical_vars.info()

# After factorization, concatenate the categorical and numerical variables

all_vars = pd.concat([categorical_vars, numerical_vars, other_vars], axis=1)
all_vars.head()
all_vars.info()
all_vars.columns = all_vars.columns.astype(str)
all_vars.info()


#### Train and Test Splitting

X_train,X_test, y_train, y_test = train_test_split(all_vars,y, test_size=0.30)

print(f"Train data shape: {X_train.shape}, Test Data Shape {X_test.shape}")


X_train.head()


### Model Training and Comparison

# Generic function that can run the passing algorithm and results the accuracy metrices
def ml_algorithm_exec(algo, X_train,y_train, cv):
    
    # Algorithm
    model = algo.fit(X_train, y_train)
    accuracy = round(model.score(X_train, y_train) * 100, 2)
    
    # Doing Cross Validation 
    pred_train = model_selection.cross_val_predict(algo,X_train,y_train,cv=cv,n_jobs = -1)
    
    # Getting Cross-validation accuracy metric
    acc_crossval = round(metrics.accuracy_score(y_train, pred_train) * 100, 2)
    
    f1 = f1_score(y_train, pred_train)
    
    return pred_train, accuracy, acc_crossval, f1


# Logistic Regression
start_time = time.time()
pred_train_log, accuracy_log, accuracy_cv_log, f1_log = ml_algorithm_exec(LogisticRegression(), X_train,y_train, 10)
log_time = (time.time() - start_time)
print("Model Score: %s" % accuracy_log)
print("Cross Validation: %s" % accuracy_cv_log)
print("F1 Score %s" % f1_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))

# SVC
start_time = time.time()
pred_train_svc, accuracy_svc, accuracy_cv_svc,f1_svc = ml_algorithm_exec(SVC(),X_train,y_train,10)
svc_time = (time.time() - start_time)
print("Model Score: %s" % accuracy_svc)
print("Cross Validation: %s" % accuracy_cv_svc)
print("F1 Score %s" % f1_svc)
print("Running Time: %s" % datetime.timedelta(seconds=svc_time))

# Linear SVC
start_time = time.time()
pred_train_linear_svc, accuracy_linear_svc, accuracy_cv_linear_svc,f1_linear_svc = ml_algorithm_exec(LinearSVC(),X_train, y_train,10)
linear_svc_time = (time.time() - start_time)
print("Model Score: %s" % accuracy_linear_svc)
print("Cross Validation: %s" % accuracy_cv_linear_svc)
print("F1 Score %s" % f1_linear_svc)
print("Running Time: %s" % datetime.timedelta(seconds=linear_svc_time))

# K Nearest Neighbour
start_time = time.time()
pred_train_knn, accuracy_knn, accuracy_cv_knn,f1_knn = ml_algorithm_exec(KNeighborsClassifier(n_neighbors = 3),X_train,y_train,10)
knn_time = (time.time() - start_time)
print("Model Score: %s" % accuracy_knn)
print("Cross Validation: %s" % accuracy_cv_knn)
print("F1 Score %s" % f1_knn)
print("Running Time: %s" % datetime.timedelta(seconds=knn_time))

# Decision Tree
start_time = time.time()
pred_train_dt, accuracy_dt, accuracy_cv_dt,f1_dt = ml_algorithm_exec(DecisionTreeClassifier(),X_train, y_train,10)
dt_time = (time.time() - start_time)
print("Model Score: %s" % accuracy_dt)
print("Cross Validation: %s" % accuracy_cv_dt)
print("F1 Score %s" % f1_dt)
print("Running Time: %s" % datetime.timedelta(seconds=dt_time))

# Random Forest
start_time = time.time()
pred_train_rf, accuracy_rf, accuracy_cv_rf,f1_rf = ml_algorithm_exec(RandomForestClassifier(n_estimators=10),X_train, y_train,10)
rf_time = (time.time() - start_time)
print("Model Score: %s" % accuracy_rf)
print("Cross Validation: %s" % accuracy_cv_rf)
print("F1 Score %s" % f1_rf)
print("Running Time: %s" % datetime.timedelta(seconds=rf_time))

# XGBoost
start_time = time.time()
pred_train_xgb, accuracy_xgb, accuracy_cv_xgb,f1_xgb = ml_algorithm_exec(XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.1),X_train, y_train,10)
xgb_time = (time.time() - start_time)
print("Model Score: %s" % accuracy_xgb)
print("Cross Validation: %s" % accuracy_cv_xgb)
print("F1 Score %s" % f1_xgb)
print("Running Time: %s" % datetime.timedelta(seconds=xgb_time))

# LightGBMBoost
start_time = time.time()
pred_train_lgb, accuracy_lgb, accuracy_cv_lgb,f1_lgb = ml_algorithm_exec(lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=50),X_train, y_train,10)
lgb_time = (time.time() - start_time)
print("Model Score: %s" % accuracy_lgb)
print("Cross Validation: %s" % accuracy_cv_lgb)
print("F1 Score %s" % f1_lgb)
print("Running Time: %s" % datetime.timedelta(seconds=lgb_time))


# ## Model Results after Training

models = pd.DataFrame({
    'Models': ['Logistic Regression','SVC','Linear SVC','KNN','Decision Tree', 'Random Forest','XGBoost','LightGBM'],
    'Result(Score)': [
        accuracy_log,
        accuracy_svc,
        accuracy_linear_svc,
        accuracy_knn, 
        accuracy_dt,
        accuracy_rf,
        accuracy_xgb,
        accuracy_lgb
    ]})
models.sort_values(by='Result(Score)', ascending=False)

cross_val_models = pd.DataFrame({
    'Model': ['Logistic Regression','SVM','Linear SVC','KNN','Decision Tree','Random Forest','XGBoost','LightGBM'],
    'Result(Cross Validation)': [
        accuracy_cv_log,
        accuracy_cv_svc,
        accuracy_cv_linear_svc,
        accuracy_cv_knn,
        accuracy_cv_dt,
        accuracy_cv_rf,
        accuracy_cv_xgb,
        accuracy_cv_lgb
    ]})
cross_val_models.sort_values(by='Result(Cross Validation)', ascending=False)

f1_models = pd.DataFrame({
    'Model': ['Logistic Regression','SVM','Linear SVC','KNN','Decision Tree','Random Forest','XGBoost','LightGBM'],
    'F1 Score': [
        f1_log,
        f1_svc,
        f1_linear_svc,
        f1_knn,
        f1_dt,
        f1_rf,
        f1_xgb,
        f1_lgb
    ]})
f1_models.sort_values(by='F1 Score', ascending=False)


# # ROC Curve

# Plotting area
plt.figure(0).clf()

# Logistic regression model for ROC curve
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Logistic Regression, AUC="+str(auc))

# SVC model for ROC curve
model = SVC(probability=True)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="SVC, AUC="+str(auc))

# LinearSVC model for ROC curve
model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.decision_function(X_test)
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="LinearSVC, AUC="+str(auc))

# KNN model for ROC curve
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="KNN, AUC="+str(auc))

# DT model for ROC curve
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="DT, AUC="+str(auc))

# RF model for ROC curve
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="RF, AUC="+str(auc))

# XGB model for ROC curve
model = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="XGB, AUC="+str(auc))

# LGB model for ROC curve
model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=50)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="LGB, AUC="+str(auc))

# legend
plt.legend()


# # Predictions and Metrices

# Function that can be run the prediction using algorithm and return the accuracy metrices
def ml_algorithm_pred_exec(algo, X_train,y_train, cv):
    model = algo.fit(X_train, y_train)
    predictions = model.predict(X_test)
    df_pred = pd.DataFrame(index=X_test.index)
    df_pred['Attrition'] = predictions
    score = round(metrics.accuracy_score(y_test, predictions) * 100, 2)
    return classification_report(y_test, predictions)

# Logistic Regression
lr_pred = ml_algorithm_pred_exec(LogisticRegression(),X_train, y_train,10)
print(lr_pred)

# SVM
svm_pred = ml_algorithm_pred_exec(SVC(),X_train, y_train,10)
print(svm_pred)

# Linear SVC
linear_svc_pred = ml_algorithm_pred_exec(LinearSVC(),X_train, y_train,10)
print(linear_svc_pred)

# KNN
knn_pred = ml_algorithm_pred_exec(KNeighborsClassifier(n_neighbors = 3),X_train, y_train,10)
print(knn_pred)

# DT
dt_pred = ml_algorithm_pred_exec(DecisionTreeClassifier(),X_train, y_train,10)
print(dt_pred)

# RF
rf_pred = ml_algorithm_pred_exec(RandomForestClassifier(n_estimators=10),X_train, y_train,10)
print(rf_pred)

# XGBoost
xgb_pred = ml_algorithm_pred_exec(XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.1),X_train, y_train,10)
print(xgb_pred)

# Light GBM
lgb_pred = ml_algorithm_pred_exec(lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=50),X_train, y_train,10)
print(lgb_pred)

# Model Ensembling
lr1 = LogisticRegression()
lr2 = LogisticRegression(random_state=42, C=0.1)
lr3 = LogisticRegression(random_state=42, C=0.2)
lr4 = LogisticRegression(random_state=50, C=0.3)
svm1 = LinearSVC()
svm2 = LinearSVC(random_state=42, C=0.1)
svm3 = LinearSVC(random_state=42, C=0.2)
svm4 = LinearSVC(random_state=50, C=0.3)

voting = VotingClassifier(estimators=[
    ('lr1', lr1),
    ('lr2', lr2),
    ('lr3', lr3),
    ('lr4', lr4),
    ('svm1', svm1),
    ('svm2', svm2),
    ('svm3', svm3),
    ('svm4', svm4)
], voting='hard')

voting.fit(X_train, y_train)

y_pred = voting.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
prec = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)

print('Voting Ensemble Results:')
print('Accuracy: {:.2f}'.format(acc))
print('Precision: {:.2f}'.format(prec))
print('Recall: {:.2f}'.format(recall))
print('F1-Score: {:.2f}'.format(f1))
print('Confusion Matrix:\n', cm)


# Logistic Regression feature Importance
importance = LogisticRegression().fit(X_train, y_train).coef_[0]

# Feature importance summarization
feature_importance = {}
for i,v in enumerate(importance):
    feature_importance[X_train.columns[i]] = v
    print('%s:  %.5f' % (X_train.columns[i], v))

# Sort the feature importance by absolute value in descending order
sorted_importance = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)

# Get the top 10 features by absolute value of their importance
top_features = dict(sorted_importance[:10])
top_features = {k: v for k, v in sorted(top_features.items(), key=lambda item: abs(item[1]), reverse=True)}

# Plot the feature importance for the top 10 positive and negative features
fig, ax = plt.subplots(figsize=(8,6))

# Plot the positive features
positive_features = {k: v for k, v in top_features.items() if v > 0}
sorted_positive = dict(sorted(positive_features.items(), key=lambda item: item[1], reverse=True))
ax.barh(list(sorted_positive.keys()), list(sorted_positive.values()), color='green')

# Plot the negative features
negative_features = {k: v for k, v in top_features.items() if v < 0}
sorted_negative = dict(sorted(negative_features.items(), key=lambda item: item[1]))
ax.barh(list(sorted_negative.keys()), list(sorted_negative.values()), color='red')

plt.title('Logistic Regression: Top 10 Positive and Negative Feature Importance Values')
plt.xlabel('Importance')
plt.ylabel('Feature')

plt.show()

# LinearSVC feature Importance
importance = LinearSVC().fit(X_train, y_train).coef_[0]

# Feature importance summarization
feature_importance = {}
for i,v in enumerate(importance):
    feature_importance[X_train.columns[i]] = v
    print('%s:  %.5f' % (X_train.columns[i], v))

# Sort the feature importance by absolute value in descending order
sorted_importance = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)

# Get the top 10 features by absolute value of their importance
top_features = dict(sorted_importance[:10])
top_features = {k: v for k, v in sorted(top_features.items(), key=lambda item: abs(item[1]), reverse=True)}

# Plot the feature importance for the top 10 positive and negative features
fig, ax = plt.subplots(figsize=(8,6))

# Plot the positive features
positive_features = {k: v for k, v in top_features.items() if v > 0}
sorted_positive = dict(sorted(positive_features.items(), key=lambda item: item[1], reverse=True))
ax.barh(list(sorted_positive.keys()), list(sorted_positive.values()), color='green')

# Plot the negative features
negative_features = {k: v for k, v in top_features.items() if v < 0}
sorted_negative = dict(sorted(negative_features.items(), key=lambda item: item[1]))
ax.barh(list(sorted_negative.keys()), list(sorted_negative.values()), color='red')

plt.title('LinearSVC: Top 10 Positive and Negative Feature Importance Values')
plt.xlabel('Importance')
plt.ylabel('Feature')

plt.show()
