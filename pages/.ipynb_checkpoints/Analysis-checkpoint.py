import pickle
import streamlit as st
st.set_page_config(page_title="Analysis", page_icon="📈",layout="wide")

import math, time, random, datetime
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from streamlit.components.v1 import html
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.tools as tls
from plotly.subplots import make_subplots
from sklearn.preprocessing import OneHotEncoder,  label_binarize, LabelEncoder, StandardScaler
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.model_selection import train_test_split,StratifiedKFold, GridSearchCV, learning_curve, cross_val_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import Perceptron,SGDClassifier,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

st.markdown("<h2 style='text-align: center; color: Black;'>Exploratory Data Analysis</h1><br/>", unsafe_allow_html=True)

df = pd.read_csv("Employee_Attrition.csv")

df.head()

df.drop(['EmployeeNumber','Over18','StandardHours','EmployeeCount'],axis=1,inplace=True)

with st.expander("Dataset Overview"):
    tab1, tab2 = st.tabs(["Overview", "Detailed Report"])
    with tab1:
        # Data source and description
        st.markdown("<h6 style='text-align: left; color: Black;'>Data Source</h6>", unsafe_allow_html=True)
        st.write("<p style='text-align: justify; color: Black;'>The data used in my analysis on employee attrition was obtained from Kaggle, a well-known platform for data science competitions, datasets, and more. Kaggle offers a wide range of datasets on various topics, and this particular dataset provides information on employee attrition in a company. With this data, I aim to analyze the factors that contribute to employee attrition and build a model to predict which employees are most likely to leave the company. For more information about Kaggle and the resources it offers for data science enthusiasts, visit https://www.kaggle.com/.</p><br/>", unsafe_allow_html=True)
        # Basic statistics
        st.markdown("<h6 style='text-align: left; color: Black;'>About Data and Features</h6>", unsafe_allow_html=True)
        col1, col2 = st.columns([1,1])
        with col1:
            num_obs = df.shape[0]
            num_feat = df.shape[1]
            missing_cells = 0
            missing_cells_percent = 0.0
            duplicates = 0
            duplicates_percent = 0.0
            num_vars = 14
            bool_vars = 2
            cat_vars = 15
            st.text("Number of Observations:\t\t\t{}\n"
                    "Number of Features:\t\t\t{}\n"
                    "Missing Cells:\t\t\t\t{}\n"
                    "Missing Cells(%):\t\t\t{}\n"
                    "Duplicate Rows:\t\t\t\t{}\n"
                    "Duplicate Rows(%):\t\t\t{}\n"
                    "Numeric Variables:\t\t\t{}\n"
                    "Boolean Variables:\t\t\t{}\n"
                    "Categorical Variables:\t\t\t{}\n".format(num_obs, num_feat, missing_cells, 
                                                               missing_cells_percent, duplicates, 
                                                               duplicates_percent, num_vars, 
                                                               bool_vars, cat_vars))

        st.write("<p style='text-align: justify; color: Black;'>This is a summary of the dataset we will be analyzing. The dataset contains 1470 observations and 31 features, representing various aspects of employee performance and demographics. There are no missing cells or duplicate rows in the dataset, indicating that the data is clean and ready for analysis. The dataset includes 14 numeric variables, such as age, monthly income, and years at company, which can be analyzed using statistical methods. Additionally, there are 2 boolean variables, such as whether the employee has overtime pay or not, and 15 categorical variables, such as job role and education level, which will require encoding before they can be used in machine learning models. This summary provides a good overview of the data we will be working with, and will help guide our analysis and modeling.</p>", unsafe_allow_html=True)
        
        st.markdown("<br/>", unsafe_allow_html=True)
        # Preview of data
        st.markdown("<h6 style='text-align: left; color: Black;'>Preview of Data</h6>", unsafe_allow_html=True)
        st.write(df.head())
        
        col_info = df.dtypes.rename_axis("Feature Name").reset_index(name="Data Type")
        col_info_list = [col_info.columns.tolist()] + col_info.to_dict("records")
        st.markdown("<p style='text-align: justify; color: black; font-size: 15px;'>All features and its datatypes:</p>", unsafe_allow_html=True)
        st.write(col_info_list, list_objects=True)
        
    with tab2:
        # Define function to generate report
        def generate_report():
            report = ProfileReport(df)
            st_profile_report(report)

        # Add button to generate report
        if st.button("Generate Report"):
            report_html = generate_report()

df['Attrition'] = df['Attrition'].apply(lambda x:1 if x == "Yes" else 0 )
df['OverTime'] = df['OverTime'].apply(lambda x:1 if x =="Yes" else 0 )

attrition = df[df['Attrition'] == 1]
no_attrition = df[df['Attrition']==0]


with st.expander("Correlation between Variables"):
    st.image("correlation.png")
    st.markdown("<p style='text-align: justify; color: black; font-size: 15px;'><i>Correlation Findings:</i></p>", unsafe_allow_html=True)
    st.markdown("* Job level is strongly correlated with total working years")
    st.markdown("* Monthly income is strongly correlated with Job level")
    st.markdown("* Monthly income is strongly correlated with total working years")
    st.markdown("* Performance Rating is strongly correlated with Salary Hike")
    st.markdown("* Department is strongly correlated with Job Role")
    st.markdown("<br/>", unsafe_allow_html=True)


def categorical_variable_viz(categ_var_name):
    f, ax = plt.subplots(1, 2, figsize=(10*0.7, 6*0.7))

    # Count Plot
    df[categ_var_name].value_counts().plot.bar(cmap='Set2', ax=ax[0])
    ax[0].set_title(f'Number of Employee by {categ_var_name}')
    ax[0].set_ylabel('Count')
    ax[0].set_xlabel(f'{categ_var_name}')

    # Attrition Count per factors
    sns.countplot(x=categ_var_name, hue='Attrition', data=df, ax=ax[1], palette='Set2')
    ax[1].set_title(f'Attrition by {categ_var_name}')
    ax[1].set_xlabel(f'{categ_var_name}')
    ax[1].set_ylabel('Count')
    st.pyplot(f)

def numerical_variable_viz(num_var_name):
    f,ax = plt.subplots(1,2, figsize=(10*0.7,7.5*0.7))
    sns.kdeplot(attrition[num_var_name], label='Employee who left',ax=ax[0], shade=True, color='palegreen')
    sns.kdeplot(no_attrition[num_var_name], label='Employee who stayed', ax=ax[0], shade=True, color='salmon')

    sns.boxplot(y=num_var_name, x='Attrition',data=df, palette='Set3', ax=ax[1])
    st.pyplot(f)

with st.expander("Analysis of Categorical Features and Numerical Features against Attrition Rate"):
    st.markdown("<br/>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2,1,2])
    with col1:
        st.markdown("<h6 style='text-align: center; color: Black;'>Categorical Features vs Attrition Rate</h6>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify; color: black; font-size: 15px;'>Plotting categorical variables against attrition rates can help identify patterns and inform strategies to reduce employee turnover. This visualization technique can highlight categories of employees most likely to leave, such as by department, job role, education, or overtime. Understanding why employees leave can guide interventions to retain them and create a stable, productive work environment.</p><br/>", unsafe_allow_html=True)
        categ_var_name = st.selectbox('', ['BusinessTravel', 'Department','EducationField','Education','EnvironmentSatisfaction','Gender',
                                                                  'JobRole','JobInvolvement','MaritalStatus','NumCompaniesWorked','OverTime','StockOptionLevel','TrainingTimesLastYear','YearsWithCurrManager'])
        with st.spinner('Loading...'):
            categorical_variable_viz(categ_var_name)

    with col3:
        st.markdown("<h6 style='text-align: center; color: Black;'>Numerical Features vs Attrition Rate</h6>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify; color: black; font-size: 15px;'>Plotting numerical variables against attrition rates can help identify patterns and inform strategies to reduce employee turnover. This visualization technique can highlight ranges of values where employees are most likely to leave, such as for salary, age, years of experience, or performance ratings. Understanding the numerical factors that contribute to employee turnover can guide interventions to retain them and create a stable, productive work environment.</p>", unsafe_allow_html=True)
        # Display numerical variable visualization in the second column
        num_var_name = st.selectbox('', ['Age', 'DailyRate','DistanceFromHome','MonthlyIncome','HourlyRate','JobInvolvement','PercentSalaryHike',
                                                                  'TotalWorkingYears','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager'])
        with st.spinner('Loading...'):
            numerical_variable_viz(num_var_name)




# ### Visualization of Categorical vs Numericals Features 
def categ_numerical(numerical_col, categorical_col1, categorical_col2):
    f, ax = plt.subplots(1, 2, figsize=(20,8))
    g1 = sns.swarmplot(x=categorical_col1, y=numerical_col, hue='Attrition', dodge=True, ax=ax[0], palette='Set2', data=df)
    ax[0].set_title(f'{numerical_col} vs {categorical_col1} separated by Attrition')
    g1.set_xticklabels(g1.get_xticklabels(), rotation=90)
    g2 = sns.swarmplot(x=categorical_col2, y=numerical_col, hue='Attrition', dodge=True, ax=ax[1], palette='Set2', data=df)
    ax[1].set_title(f'{numerical_col} vs {categorical_col2} separated by Attrition')
    g2.set_xticklabels(g2.get_xticklabels(), rotation=90)
    st.pyplot(f)

with st.expander("Analysis of Numerical Features against Categorical Features (Attrition Rate)"):
    col1, col2, col3 = st.columns([2,2,2])
    with col1:
        numerical_col = st.selectbox('Numerical Feature', ['Age', 'DailyRate','DistanceFromHome','MonthlyIncome','HourlyRate','JobInvolvement','PercentSalaryHike',
                                                                  'TotalWorkingYears','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager'])
    with col2:
        categorical_col1 = st.selectbox('Categorical Feature 1', ['BusinessTravel', 'Department','EducationField','Education','EnvironmentSatisfaction','Gender',
                                                                  'JobRole','JobInvolvement','MaritalStatus','NumCompaniesWorked','OverTime','StockOptionLevel','TrainingTimesLastYear','YearsWithCurrManager'])
    with col3:
        categorical_col2 = st.selectbox('Categorical Feature 2', ['BusinessTravel', 'Department','EducationField','Education','EnvironmentSatisfaction','Gender',
                                                                  'JobRole','JobInvolvement','MaritalStatus','NumCompaniesWorked','OverTime','StockOptionLevel','TrainingTimesLastYear','YearsWithCurrManager'])
    col1, col2, col3 = st.columns([1,6,1])
    with col2:
        with st.spinner('Loading...'):
            categ_numerical(numerical_col, categorical_col1, categorical_col2)



# 'EnviornmentSatisfaction', 'JobInvolvement', 'JobSatisfacction', 'RelationshipSatisfaction', 'WorklifeBalance' can be joined together into a single variable 'TotalSatisfaction'

df['Total_Satisfaction'] = (df['EnvironmentSatisfaction'] + 
                            df['JobInvolvement'] + 
                            df['JobSatisfaction'] + 
                            df['RelationshipSatisfaction'] +
                            df['WorkLifeBalance']) /5 

# Drop the variables after clubbing
df.drop(['EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance'], axis=1, inplace=True)

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

# We are separating the categorical and numerical data
X_categorical = df.select_dtypes(include=['category'])
X_numerical = df.select_dtypes(include=['int64'])
X_other = df.select_dtypes(include=['float64'])

# Drop the attrition variable as that is dependant variable
X_numerical.drop('Attrition', axis=1, inplace=True)

y = df['Attrition']

# Doing factozation for the categorical variables
X_categorical = pd.get_dummies(X_categorical)


# After factorization, concatenate the categorical and numerical variables
X_all = pd.concat([X_categorical, X_numerical, X_other], axis=1)
X_all.columns = X_all.columns.astype(str)

# ### Split the Dataset into Train and Test
X_train,X_test, y_train, y_test = train_test_split(X_all,y, test_size=0.30)

# ## Model Training and Comparison
# Generic function that can run the passing algorithm and results the accuracy metrics
def exec_ml_algorithm(algo, X_train,y_train, cv):

    # Algorithm
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)

    # Doing Cross Validation 
    train_pred = model_selection.cross_val_predict(algo,X_train,y_train,cv=cv,n_jobs = -1)

    # Getting Cross-validation accuracy metric
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)

    return train_pred, acc, acc_cv

with st.expander("Model Comparison"):
    col1, col2 = st.columns([4,4])
    with col1:
        st.markdown("<br/>", unsafe_allow_html=True)
        algo = st.selectbox('Select the Algorithm', [LogisticRegression(), SVC(),LinearSVC(),KNeighborsClassifier(n_neighbors = 3),DecisionTreeClassifier(),RandomForestClassifier(n_estimators=100)])

    
    start_time = time.time()
    train_pred_log, acc_log, acc_cv_log = exec_ml_algorithm(algo, X_train,y_train, 10)
    log_time = round((time.time() - start_time),5)
    col1, col2, col3 = st.columns(3)
    with st.spinner('Loading...'):
        col1.metric("Accuracy", str(acc_log), "")
        col2.metric("Accuracy CV 10-Fold", str(acc_cv_log), "")
        col3.metric("Running Time", str(log_time), "")
