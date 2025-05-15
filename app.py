import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, load_model



df = load_data()
model = load_model()



# Sidebar Filters
st.sidebar.header("Filter Employees by Key Features")

project_range = st.sidebar.slider("Number of Projects", 1, 10, (2, 6))
tenure_range = st.sidebar.slider("Tenure", 1, 10, (2, 6))
overworked_only = st.sidebar.checkbox("Overworked", value=False)#i.e  (Hours > 250)
department_filter = st.sidebar.multiselect("Department", df['department'].unique())
salary_filter = st.sidebar.multiselect("Salary Level", df['salary'].unique())

filtered_df = df[
    (df['number_project'] >= project_range[0]) &
    (df['number_project'] <= project_range[1]) &
    (df['tenure'] >= tenure_range[0]) &
    (df['tenure'] <= tenure_range[1])
]


if overworked_only:
    filtered_df = filtered_df[filtered_df['average_monthly_hours'] > 250]

if department_filter:
    filtered_df = filtered_df[filtered_df['department'].isin(department_filter)]
    
if salary_filter:
    filtered_df = filtered_df[filtered_df['salary'].isin(salary_filter)]


# Key Metrics
st.subheader(" Key Insights")
col1, col2, col3 = st.columns(3)
col1.metric("Attrition Rate", f"{filtered_df['left'].mean() * 100:.2f}%")
col2.metric("Avg Satisfaction", f"{filtered_df['satisfaction_level'].mean():.2f}")
col3.metric("Top Risk Dept.", filtered_df.groupby('department')['left'].mean().idxmax())



st.subheader("Visual HR Analysis")
tab1, tab2, tab3, tab4,tab5 = st.tabs(["Stayed/Left by Dept", "Satisfaction Trend", "Promotion Impact", "Heatmap","Satisfaction and Tenure Analysis"])

with tab1:
    fig = plt.figure(figsize=(12, 6))
    sns.histplot(
        data=filtered_df, 
        x='department', 
        hue='left', 
        discrete=True,
        hue_order=[0, 1],
        multiple='dodge',
        shrink=0.5
    )
    plt.xticks(rotation=45)
    plt.title('Counts of Stayed vs Left by department', fontsize=14)
    plt.xlabel("Department")
    plt.ylabel("Count")
    st.pyplot(fig)

    
    

with tab2:
    fig, ax = plt.subplots()
    filtered_df.groupby('tenure')['satisfaction_level'].mean().plot(kind='line', marker='o', ax=ax)
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

    sns.boxplot(
        data=filtered_df, 
        x='satisfaction_level', 
        y='tenure', 
        hue='left', 
        orient="h", 
        ax=ax[0]
    )
    ax[0].invert_yaxis()
    ax[0].set_title('Satisfaction by Tenure')
    ax[0].set_xlabel("Satisfaction Level")
    ax[0].set_ylabel("Years at Company")



    sns.histplot(
        data=filtered_df, 
        x='tenure', 
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
    
    
    
# High Risk Employee List
st.subheader("High-Risk Employee List")
risk_df = df[(df['satisfaction_level'] < 0.4) & (df['average_monthly_hours'] > 250)]
st.dataframe(risk_df[['department', 'number_project','tenure','satisfaction_level','salary' ,'average_monthly_hours', 'left']])
st.download_button("Download High-Risk Data", risk_df.to_csv(index=False), "high_risk_employees.csv")


# Prediction form section 
st.subheader("Predict Attrition")
with st.form("predict_form"):
  
    satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    last_eval = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
    projects = st.number_input("Number of Projects", 1, 10, 3)
    hours = st.slider("Average Monthly Hours", 80, 320, 160)
    tenure = st.slider("Years at Company", 1, 10, 3)
    department = st.selectbox("Department", df['department'].unique())
    salary = st.selectbox("Salary Level", df['salary'].unique())

    submitted = st.form_submit_button("Predict")
    if submitted:
        input_data = {
            'satisfaction_level': satisfaction,
            'last_evaluation': last_eval,
            'number_project': projects,
            'average_monthly_hours': hours,
            'tenure': tenure,
            'Work_accident': 0,
            'promotion_last_5years': 0,
            'department': department,
            'salary': salary
        }
        input_df = pd.DataFrame([input_data])

        # Encode the 'salary' column as ordinal
        input_df['salary'] = (
            input_df['salary'].astype('category')
            .cat.set_categories(['low', 'medium', 'high'])
            .cat.codes
        )

        input_encoded = pd.get_dummies(input_df, drop_first=False)

        # Ensure all columns used in the model are present
        missing_cols = set(model.feature_names_in_) - set(input_encoded.columns)
        # print(f"Missing columns: {missing_cols}")
        for col in missing_cols:
            input_encoded[col] = 0  # Add missing columns with 0s

        input_encoded = input_encoded[model.feature_names_in_]
        st.write(input_encoded)

        prediction_probability = model.predict_proba(input_encoded)[0][1]
        prediction = model.predict(input_encoded)[0]
        
        st.success(f"Probability of Leaving: {prediction_probability:.2%}")

        if prediction_probability >= 0.7:
            st.error("High Risk: This employee is very likely to leave.")
        elif 0.3 <= prediction_probability < 0.7:
            st.warning("Medium Risk: This employee might leave. Monitor closely.")
        else:
            st.success("Low Risk: This employee is likely to stay.")

        
     