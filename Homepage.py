import streamlit as st

st.set_page_config(page_title="Employee Attrition Analysis", page_icon=":guardsman:", layout="wide")

    
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.haleymarketing.com/wp-content/uploads/2020/03/671784_API-MLW-Haley-Marketing-Facebook-Social-Sharing-Images-for-Q1_06-v2_031820.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

col1, col2, col3 = st.columns([4,2,2])
with col1:
    # Add header and subheading
    st.markdown("<br/><br/><br/><h1 style='text-align: center; color: black; font-size: 60px; text-shadow: 2px 2px #FFFFFF;'>Employee Attrition Analysis and Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: black; font-size: 30px; text-shadow: 2px 2px #FFFFFF;'>Exploring Employee Turnover using Machine Learning</h3>", unsafe_allow_html=True)

    # Add main text and image
    st.markdown("<br/><p style='text-align: justify; color: black; font-size: 15px;'>Welcome to my academic project focused on exploring employee attrition using machine learning. In this project, I will be analyzing data related to employee turnover to identify patterns, trends, and factors that contribute to employee attrition and utilizing various machine learning techniques to develop predictive models that can help organizations identify employees who are at risk of leaving, and take proactive measures to retain them.</p><br/>", unsafe_allow_html=True)

    # Add call-to-action button
    st.markdown("<div style='text-align:center;'><a href='/Analysis' style='font-size: 24px; background-color: #0072b0; color: #FFFFFF; padding: 15px 50px; border-radius: 10px; text-decoration: none; box-shadow: 2px 2px #000000;'>Go to Analysis</a></div>", unsafe_allow_html=True)


