import numpy as np
import scipy as sp
import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="Prediction", page_icon="📈",layout="wide")

model = pickle.load (open ('model.pkl','rb'))

model_features = model.feature_names_in_

st.markdown("<h3 style='text-align: center; color: black;'>Employee Attrition Prediction</h3>", unsafe_allow_html=True)
with st.form("attrition_form"):
    with st.expander("ℹ️ Please provide some Personal information about the Employee"):
        col1, col2 = st.columns(2)
        with col1:
            Age = st.slider("Age", min_value=18, max_value=70, value=29, step=1)
            Gender = st.selectbox('Gender',('Male', 'Female'))
            MaritalStatus = st.selectbox('Marital Status',('Single', 'Married', 'Divorced'))
            DistanceFromHome = st.slider("DistanceFromHome", min_value=1, max_value=100, value=5, step=1)
            TotalWorkingYears = st.slider("TotalWorkingYears", min_value=0, max_value=20, value=4, step=1)
            NumCompaniesWorked = st.slider("NumCompaniesWorked", min_value=1, max_value=15, value=3, step=1)
        with col2:
            OverTime = st.selectbox('OverTime',('Yes', 'No'))
            BusinessTravel = st.selectbox('Business Travel',('Rarely', 'Frequently', 'Non-Travel'))
            Department = st.selectbox('Department',('Research & Development', 'Sales', 'Other'))
            Education = st.selectbox('Education',('Below College', 'College', 'Bachelor', 'Master', 'Doctor'))
            YearsAtCompany = st.slider("YearsAtCompany", min_value=1, max_value=20, value=4, step=1)
            EducationField = st.selectbox('EducationField',('Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'))
    
    with st.expander("ℹ️ Please provide some Job related information about the Employee"):
        col1, col2 = st.columns(2)
        with col1:
            JobLevel = st.slider("JobLevel", min_value=2, max_value=5, value=1, step=1) 
            HourlyRate = st.text_input("HourlyRate (£)",value=94)
            DailyRate = st.text_input("DailyRate (£)",value=1102)
            MonthlyIncome = st.text_input("MonthlyIncome (£)",value=5993)
            YearsInCurrentRole = st.slider("YearsInCurrentRole", min_value=1, max_value=20, value=2, step=1)
            YearsWithCurrManager = st.slider("YearsWithCurrManager", min_value=1, max_value=20, value=4, step=1) 
            YearsSinceLastPromotion = st.slider("YearsSinceLastPromotion", min_value=1, max_value=20, value=2, step=1)
            TrainingTimesLastYear = st.slider("TrainingTimesLastYear", min_value=1, max_value=20, value=3, step=1)
        with col2:
            JobRole = st.selectbox('JobRole',('Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative'))
            PerformanceRating = st.slider("PerformanceRating", min_value=1, max_value=4, value=2, step=1)
            EnvironmentSatisfaction = st.slider("Environment Satisfaction", min_value=1, max_value=4, value=2, step=1)
            JobInvolvement = st.slider("Job Involvement", min_value=1, max_value=4, value=3, step=1)
            JobSatisfaction = st.slider("Job Satisfaction", min_value=1, max_value=4, value=4, step=1)
            RelationshipSatisfaction = st.slider("Relationship Satisfaction", min_value=1, max_value=4, value=1, step=1)
            WorkLifeBalance = st.slider("Work-Life Balance", min_value=1, max_value=4, value=1, step=1)
            StockOptionLevel = st.slider("StockOptionLevel", min_value=1, max_value=1, value=4, step=1)      
    col1, col2, col3= st.columns(3) 
    with col2:
        submitted = st.form_submit_button("Predict")

    
    dict = {
        'Age': int (Age),
        'BusinessTravel': str (BusinessTravel),
        'DailyRate': int (DailyRate),
        'Department': Department,
        'DistanceFromHome': int (DistanceFromHome),
        'Education': Education,
        'EducationField': str (EducationField),
        'EnvironmentSatisfaction': int (EnvironmentSatisfaction),
        'Gender': str (Gender),
        'HourlyRate': int (HourlyRate),
        'JobInvolvement': int (JobInvolvement),
        'JobLevel': int (JobLevel),
        'JobRole': JobRole,
        'JobSatisfaction': int (JobSatisfaction),
        'MaritalStatus': str (MaritalStatus),
        'MonthlyIncome': int (MonthlyIncome),
        'NumCompaniesWorked': int (NumCompaniesWorked),
        'OverTime': str (OverTime),
        'PerformanceRating': int (PerformanceRating),
        'RelationshipSatisfaction': int (RelationshipSatisfaction),
        'StockOptionLevel': StockOptionLevel,
        'TotalWorkingYears': int (TotalWorkingYears),
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'WorkLifeBalance': int (WorkLifeBalance),
        'YearsAtCompany': int (YearsAtCompany),
        'YearsInCurrentRole': int (YearsInCurrentRole),
        'YearsSinceLastPromotion': int (YearsSinceLastPromotion),
        'YearsWithCurrManager': int (YearsWithCurrManager)
    }
    
    df = pd.DataFrame ([dict])
    
    df['Total_Satisfaction'] = (df['EnvironmentSatisfaction'] +
                                df['JobInvolvement'] +
                                df['JobSatisfaction'] +
                                df['RelationshipSatisfaction'] +
                                df['WorkLifeBalance']) / 5

    # Drop Satisfaction Columns
    df.drop (
        ['EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance'],
        axis=1,inplace=True)

    # Convert Total satisfaction into boolean
    df['Total_Satisfaction_bool'] = df['Total_Satisfaction'].apply (lambda x: 1 if x >= 2.8 else 0)
    df.drop ('Total_Satisfaction',axis=1,inplace=True)

    # It can be observed that the rate of attrition of employees below age of 35 is high
    df['Age_bool'] = df['Age'].apply (lambda x: 1 if x < 35 else 0)
    df.drop ('Age',axis=1,inplace=True)

    # It can be observed that the employees are more likey the drop the job if dailyRate less than 800
    df['DailyRate_bool'] = df['DailyRate'].apply (lambda x: 1 if x < 800 else 0)
    df.drop ('DailyRate',axis=1,inplace=True)

    # Employees working at R&D Department have higher attrition rate
    df['Department_bool'] = df['Department'].apply (lambda x: 1 if x == 'Research & Development' else 0)
    df.drop ('Department',axis=1,inplace=True)

    # Rate of attrition of employees is high if DistanceFromHome > 10
    df['DistanceFromHome_bool'] = df['DistanceFromHome'].apply (lambda x: 1 if x > 10 else 0)
    df.drop ('DistanceFromHome',axis=1,inplace=True)

    # Employees are more likey to drop the job if the employee is working as Laboratory Technician
    df['JobRole_bool'] = df['JobRole'].apply (lambda x: 1 if x == 'Laboratory Technician' else 0)
    df.drop ('JobRole',axis=1,inplace=True)

    # Employees are more likey to the drop the job if the employee's hourly rate < 65
    df['HourlyRate_bool'] = df['HourlyRate'].apply (lambda x: 1 if x < 65 else 0)
    df.drop ('HourlyRate',axis=1,inplace=True)

    # Employees are more likey to the drop the job if the employee's MonthlyIncome < 4000
    df['MonthlyIncome_bool'] = df['MonthlyIncome'].apply (lambda x: 1 if x < 4000 else 0)
    df.drop ('MonthlyIncome',axis=1,inplace=True)

    # Rate of attrition of employees is high if NumCompaniesWorked < 3
    df['NumCompaniesWorked_bool'] = df['NumCompaniesWorked'].apply (lambda x: 1 if x > 3 else 0)
    df.drop ('NumCompaniesWorked',axis=1,inplace=True)

    # Employees are more likey to the drop the job if the employee's TotalWorkingYears < 8
    df['TotalWorkingYears_bool'] = df['TotalWorkingYears'].apply (lambda x: 1 if x < 8 else 0)
    df.drop ('TotalWorkingYears',axis=1,inplace=True)

    # Employees are more likey to the drop the job if the employee's YearsAtCompany < 3
    df['YearsAtCompany_bool'] = df['YearsAtCompany'].apply (lambda x: 1 if x < 3 else 0)
    df.drop ('YearsAtCompany',axis=1,inplace=True)

    # Employees are more likey to the drop the job if the employee's YearsInCurrentRole < 3
    df['YearsInCurrentRole_bool'] = df['YearsInCurrentRole'].apply (lambda x: 1 if x < 3 else 0)
    df.drop ('YearsInCurrentRole',axis=1,inplace=True)

    # Employees are more likely to the drop the job if the employee's YearsSinceLastPromotion < 1
    df['YearsSinceLastPromotion_bool'] = df['YearsSinceLastPromotion'].apply (lambda x: 1 if x < 1 else 0)
    df.drop ('YearsSinceLastPromotion',axis=1,inplace=True)

    # Employees are more likely to the drop the job if the employee's YearsWithCurrManager < 1
    df['YearsWithCurrManager_bool'] = df['YearsWithCurrManager'].apply (lambda x: 1 if x < 1 else 0)
    df.drop ('YearsWithCurrManager',axis=1,inplace=True)

    # Convert Categorical to Numerical
    # Buisness Travel
    if BusinessTravel == 'Rarely':
        df['BusinessTravel_Travel_Rarely'] = 1
        df['BusinessTravel_Travel_Frequently'] = 0
        df['BusinessTravel_Non-Travel'] = 0
    elif BusinessTravel == 'Frequently':
        df['BusinessTravel_Travel_Rarely'] = 0
        df['BusinessTravel_Travel_Frequently'] = 1
        df['BusinessTravel_Non-Travel'] = 0
    else:
        df['BusinessTravel_Travel_Rarely'] = 0
        df['BusinessTravel_Travel_Frequently'] = 0
        df['BusinessTravel_Non-Travel'] = 1
    df.drop ('BusinessTravel',axis=1,inplace=True)

    # Education
    if Education == 'Below College':
        df['Education_1'] = 1
        df['Education_2'] = 0
        df['Education_3'] = 0
        df['Education_4'] = 0
        df['Education_5'] = 0
    elif Education == 'College':
        df['Education_1'] = 0
        df['Education_2'] = 1
        df['Education_3'] = 0
        df['Education_4'] = 0
        df['Education_5'] = 0
    elif Education == 'Bachelor':
        df['Education_1'] = 0
        df['Education_2'] = 0
        df['Education_3'] = 1
        df['Education_4'] = 0
        df['Education_5'] = 0
    elif Education == 'Master':
        df['Education_1'] = 0
        df['Education_2'] = 0
        df['Education_3'] = 0
        df['Education_4'] = 1
        df['Education_5'] = 0
    else:
        df['Education_1'] = 0
        df['Education_2'] = 0
        df['Education_3'] = 0
        df['Education_4'] = 0
        df['Education_5'] = 1
    df.drop ('Education',axis=1,inplace=True)

    # EducationField
    if EducationField == 'Life Sciences':
        df['EducationField_Life Sciences'] = 1
        df['EducationField_Medical'] = 0
        df['EducationField_Marketing'] = 0
        df['EducationField_Technical Degree'] = 0
        df['EducationField_Human Resources'] = 0
        df['EducationField_Other'] = 0
    elif EducationField == 'Medical':
        df['EducationField_Life Sciences'] = 0
        df['EducationField_Medical'] = 1
        df['EducationField_Marketing'] = 0
        df['EducationField_Technical Degree'] = 0
        df['EducationField_Human Resources'] = 0
        df['EducationField_Other'] = 0
    elif EducationField == 'Marketing':
        df['EducationField_Life Sciences'] = 0
        df['EducationField_Medical'] = 0
        df['EducationField_Marketing'] = 1
        df['EducationField_Technical Degree'] = 0
        df['EducationField_Human Resources'] = 0
        df['EducationField_Other'] = 0
    elif EducationField == 'Technical Degree':
        df['EducationField_Life Sciences'] = 0
        df['EducationField_Medical'] = 0
        df['EducationField_Marketing'] = 0
        df['EducationField_Technical Degree'] = 1
        df['EducationField_Human Resources'] = 0
        df['EducationField_Other'] = 0
    elif EducationField == 'Human Resources':
        df['EducationField_Life Sciences'] = 0
        df['EducationField_Medical'] = 0
        df['EducationField_Marketing'] = 0
        df['EducationField_Technical Degree'] = 0
        df['EducationField_Human Resources'] = 1
        df['EducationField_Other'] = 0
    else:
        df['EducationField_Life Sciences'] = 0
        df['EducationField_Medical'] = 0
        df['EducationField_Marketing'] = 0
        df['EducationField_Technical Degree'] = 0
        df['EducationField_Human Resources'] = 1
        df['EducationField_Other'] = 1
    df.drop ('EducationField',axis=1,inplace=True)

    # Gender
    if Gender == 'Male':
        df['Gender_0'] = 1
        df['Gender_1'] = 0
    else:
        df['Gender_0'] = 0
        df['Gender_1'] = 1
    df.drop ('Gender',axis=1,inplace=True)

    # Marital Status
    if MaritalStatus == 'Married':
        df['MaritalStatus_Married'] = 1
        df['MaritalStatus_Single'] = 0
        df['MaritalStatus_Divorced'] = 0
    elif MaritalStatus == 'Single':
        df['MaritalStatus_Married'] = 0
        df['MaritalStatus_Single'] = 1
        df['MaritalStatus_Divorced'] = 0
    else:
        df['MaritalStatus_Married'] = 0
        df['MaritalStatus_Single'] = 0
        df['MaritalStatus_Divorced'] = 1
    df.drop ('MaritalStatus',axis=1,inplace=True)

    # Overtime
    if OverTime == 'Yes':
        df['OverTime_0'] = 1
        df['OverTime_1'] = 0
    else:
        df['OverTime_0'] = 0
        df['OverTime_1'] = 1
    df.drop ('OverTime',axis=1,inplace=True)

    # Stock Option Level
    if StockOptionLevel == 0:
        df['StockOptionLevel_0'] = 1
        df['StockOptionLevel_1'] = 0
        df['StockOptionLevel_2'] = 0
        df['StockOptionLevel_3'] = 0
    elif StockOptionLevel == 1:
        df['StockOptionLevel_0'] = 0
        df['StockOptionLevel_1'] = 1
        df['StockOptionLevel_2'] = 0
        df['StockOptionLevel_3'] = 0
    elif StockOptionLevel == 2:
        df['StockOptionLevel_0'] = 0
        df['StockOptionLevel_1'] = 0
        df['StockOptionLevel_2'] = 1
        df['StockOptionLevel_3'] = 0
    else:
        df['StockOptionLevel_0'] = 0
        df['StockOptionLevel_1'] = 0
        df['StockOptionLevel_2'] = 0
        df['StockOptionLevel_3'] = 1
    df.drop ('StockOptionLevel',axis=1,inplace=True)

    # Training Time Last Year
    if TrainingTimesLastYear == 0:
        df['TrainingTimesLastYear_0'] = 1
        df['TrainingTimesLastYear_1'] = 0
        df['TrainingTimesLastYear_2'] = 0
        df['TrainingTimesLastYear_3'] = 0
        df['TrainingTimesLastYear_4'] = 0
        df['TrainingTimesLastYear_5'] = 0
        df['TrainingTimesLastYear_6'] = 0
    elif TrainingTimesLastYear == 1:
        df['TrainingTimesLastYear_0'] = 0
        df['TrainingTimesLastYear_1'] = 1
        df['TrainingTimesLastYear_2'] = 0
        df['TrainingTimesLastYear_3'] = 0
        df['TrainingTimesLastYear_4'] = 0
        df['TrainingTimesLastYear_5'] = 0
        df['TrainingTimesLastYear_6'] = 0
    elif TrainingTimesLastYear == 2:
        df['TrainingTimesLastYear_0'] = 0
        df['TrainingTimesLastYear_1'] = 0
        df['TrainingTimesLastYear_2'] = 1
        df['TrainingTimesLastYear_3'] = 0
        df['TrainingTimesLastYear_4'] = 0
        df['TrainingTimesLastYear_5'] = 0
        df['TrainingTimesLastYear_6'] = 0
    elif TrainingTimesLastYear == 3:
        df['TrainingTimesLastYear_0'] = 0
        df['TrainingTimesLastYear_1'] = 0
        df['TrainingTimesLastYear_2'] = 0
        df['TrainingTimesLastYear_3'] = 1
        df['TrainingTimesLastYear_4'] = 0
        df['TrainingTimesLastYear_5'] = 0
        df['TrainingTimesLastYear_6'] = 0
    elif TrainingTimesLastYear == 4:
        df['TrainingTimesLastYear_0'] = 0
        df['TrainingTimesLastYear_1'] = 0
        df['TrainingTimesLastYear_2'] = 0
        df['TrainingTimesLastYear_3'] = 0
        df['TrainingTimesLastYear_4'] = 1
        df['TrainingTimesLastYear_5'] = 0
        df['TrainingTimesLastYear_6'] = 0
    elif TrainingTimesLastYear == 5:
        df['TrainingTimesLastYear_0'] = 0
        df['TrainingTimesLastYear_1'] = 0
        df['TrainingTimesLastYear_2'] = 0
        df['TrainingTimesLastYear_3'] = 0
        df['TrainingTimesLastYear_4'] = 0
        df['TrainingTimesLastYear_5'] = 1
        df['TrainingTimesLastYear_6'] = 0
    else:
        df['TrainingTimesLastYear_0'] = 0
        df['TrainingTimesLastYear_1'] = 0
        df['TrainingTimesLastYear_2'] = 0
        df['TrainingTimesLastYear_3'] = 0
        df['TrainingTimesLastYear_4'] = 0
        df['TrainingTimesLastYear_5'] = 0
        df['TrainingTimesLastYear_6'] = 1
    df.drop ('TrainingTimesLastYear',axis=1,inplace=True)
    

    # df.to_csv ('features.csv',index=False)

    prediction = model.predict (df[model_features])

    if submitted:
        if prediction == 0:
            st.markdown("<h4 style='text-align: center; color: black;'><i>Employee Might Not Leave The Job</i></h4>", unsafe_allow_html=True)
        else:
            st.markdown("<h4 style='text-align: center; color: black;'><i>Employee Might Leave The Job</i></h4>", unsafe_allow_html=True)

