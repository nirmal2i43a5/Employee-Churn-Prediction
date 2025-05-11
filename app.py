import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv('dataset.csv')

@st.cache_resource
def load_model():
    
    with open('models/hr_rf2.pickle', 'rb') as f:
        return pickle.load(f)

df = load_data()
model = load_model()


'''Sidebar Filters'''
st.sidebar.header("Filter Employees by Key Features")

project_range = st.sidebar.slider("Number of Projects", 1, 10, (2, 6))
tenure_range = st.sidebar.slider("Years at Company", 1, 10, (2, 6))
overworked_only = st.sidebar.checkbox("Overworked", value=False)#i.e  (Hours > 250)
department_filter = st.sidebar.multiselect("Department", df['Department'].unique())
salary_filter = st.sidebar.multiselect("Salary Level", df['salary'].unique())
filtered_df = df[
    (df['number_project'] >= project_range[0]) &
    (df['number_project'] <= project_range[1]) &
    (df['time_spend_company'] >= tenure_range[0]) &
    (df['time_spend_company'] <= tenure_range[1])
]

if overworked_only:
    filtered_df = filtered_df[filtered_df['average_montly_hours'] > 250]

if department_filter:
    filtered_df = filtered_df[filtered_df['Department'].isin(department_filter)]
if salary_filter:
    filtered_df = filtered_df[filtered_df['salary'].isin(salary_filter)]


'''Key Metrics'''
st.subheader("📊 Key Insights")
col1, col2, col3 = st.columns(3)
col1.metric("Attrition Rate", f"{filtered_df['left'].mean() * 100:.2f}%")
col2.metric("Avg Satisfaction", f"{filtered_df['satisfaction_level'].mean():.2f}")
col3.metric("Top Risk Dept.", filtered_df.groupby('Department')['left'].mean().idxmax())



st.subheader("Visual HR Analysis")
tab1, tab2, tab3, tab4,tab5 = st.tabs(["Stayed/Left by Dept", "Satisfaction Trend", "Promotion Impact", "Heatmap","Satisfaction and Tenure Analysis"])

with tab1:
    fig = plt.figure(figsize=(12, 6))
    sns.histplot(
        data=filtered_df, 
        x='Department', 
        hue='left', 
        discrete=True,
        hue_order=[0, 1],
        multiple='dodge',
        shrink=0.5
    )
    plt.xticks(rotation=45)
    plt.title('Counts of Stayed vs Left by Department', fontsize=14)
    plt.xlabel("Department")
    plt.ylabel("Count")
    st.pyplot(fig)

    
    

with tab2:
    fig, ax = plt.subplots()
    filtered_df.groupby('time_spend_company')['satisfaction_level'].mean().plot(kind='line', marker='o', ax=ax)
    ax.set_title("Satisfaction Trend by Tenure")
    ax.set_ylabel("Avg Satisfaction")
    st.pyplot(fig)

with tab3:
  
    st.markdown("### Promotion Impact on Attrition")
    promo_df = pd.crosstab(df['promotion_last_5years'], df['left'], normalize='index')
    fig2, ax2 = plt.subplots()
    promo_df.plot(kind='bar', ax=ax2, stacked=True)
    ax2.set_title("Attrition Rate by Promotion in Last 5 Years")
    ax2.set_ylabel("Proportion")
    ax2.legend(["Stayed", "Left"])
    st.pyplot(fig2)

with tab4:
    fig, ax = plt.subplots()
    corr = filtered_df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)

with tab5:
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 16))

    # Creating boxplot showing distributions of `satisfaction_level` by tenure, comparing employees who stayed versus those who left
    sns.boxplot(
        data=filtered_df, 
        x='satisfaction_level', 
        y='time_spend_company', 
        hue='left', 
        orient="h", 
        ax=ax[0]
    )
    ax[0].invert_yaxis()
    ax[0].set_title('Satisfaction by Tenure')
    ax[0].set_xlabel("Satisfaction Level")
    ax[0].set_ylabel("Years at Company")


    # Creating histogram showing distribution of `tenure`, comparing employees who stayed versus those who left

    sns.histplot(
        data=filtered_df, 
        x='time_spend_company', 
        hue='left', 
        multiple='dodge', 
        shrink=0.8, 
        discrete=True,
        ax=ax[1]
    )
    ax[1].set_title('Tenure Histogram (Stayed vs Left)')
    ax[1].set_xlabel("Years at Company")
    ax[1].set_ylabel("Count")

    st.pyplot(fig)
# ---------------------
# High-Risk Table & Export
# ---------------------
st.subheader("High-Risk Employee List")
risk_df = df[(df['satisfaction_level'] < 0.4) & (df['average_montly_hours'] > 250)]
st.dataframe(risk_df[['Department', 'satisfaction_level', 'average_montly_hours', 'left']])
st.download_button("Download High-Risk Data", risk_df.to_csv(index=False), "high_risk_employees.csv")

# ---------------------
# Prediction Section
# ---------------------
st.subheader("Predict Attrition")
with st.form("predict_form"):
    satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    last_eval = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
    projects = st.number_input("Number of Projects", 1, 10, 3)
    hours = st.slider("Average Monthly Hours", 80, 320, 160)
    tenure = st.slider("Years at Company", 1, 10, 3)
    accident = st.selectbox("Had Work Accident?", [0, 1])
    promotion = st.selectbox("Promoted in Last 5 Years?", [0, 1])
    department = st.selectbox("Department", df['Department'].unique())
    salary = st.selectbox("Salary Level", df['salary'].unique())

    submitted = st.form_submit_button("Predict")
    if submitted:
        input_data = {
            'satisfaction_level': satisfaction,
            'last_evaluation': last_eval,
            'number_project': projects,
            'average_montly_hours': hours,
            'time_spend_company': tenure,
            'Work_accident': accident,
            'promotion_last_5years': promotion,
            'Department': department,
            'salary': salary
        }

        input_df = pd.DataFrame([input_data])
        
        input_encoded = pd.get_dummies(input_df)
        for col in model.feature_names_in_:
            if col not in input_encoded:
                input_encoded[col] = 0
        input_encoded = input_encoded[model.feature_names_in_]

        prediction = model.predict_proba(input_encoded)[0][1]
        st.success(f"Probability of Leaving: {prediction:.2%}")


