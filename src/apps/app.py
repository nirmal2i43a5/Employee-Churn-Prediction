import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_data, load_pipeline
from predict import predict_form
from eda import show_eda



def main():
    
    df = load_data()
    pipeline = load_pipeline()

    # Sidebar Filters
    st.sidebar.header("Filter Employees by Key Features")
    satisfaction_range = st.sidebar.slider(
        "Satisfaction Level Range", 0.0, 1.0, (0.3, 0.8)
    )
    project_range = st.sidebar.slider(
        "Number of Projects", 1, 10, (2, 6)
    )
    tenure_range = st.sidebar.slider(
        "Tenure (Years)", 1, 10, (2, 6)
    )
    overworked_only = st.sidebar.checkbox(
        "Overworked Only (Hours > 175)", value=False
    )
    department_filter = st.sidebar.multiselect(
        "Department", df['department'].unique()
    )
    salary_filter = st.sidebar.multiselect(
        "Salary Level", df['salary'].unique()
    )

    # Apply filters
    filtered_df = df[
        (df['satisfaction_level'] >= satisfaction_range[0]) &
        (df['satisfaction_level'] <= satisfaction_range[1]) &
        
        (df['number_project'] >= project_range[0]) &
        (df['number_project'] <= project_range[1]) &
        
        (df['tenure'] >= tenure_range[0]) &
        (df['tenure'] <= tenure_range[1])
    ]

    if overworked_only:
        filtered_df = filtered_df[filtered_df['average_monthly_hours'] > 175]

    if department_filter:
        filtered_df = filtered_df[filtered_df['department'].isin(department_filter)]

    if salary_filter:
        filtered_df = filtered_df[filtered_df['salary'].isin(salary_filter)]

    # Key Metrics
    st.subheader(" Key Insights")
    # prediction_probability = pipeline.predict_proba(filtered_df)[0][1]
    col1, col2, col3 = st.columns(3)
    # col1.metric("Avg Attrition Rate", f"{prediction_probability} * 100%")
    col1.metric("Avg Attrition Rate", f"{filtered_df['left'].mean() * 100:.2f}%")
    
    col2.metric("Avg Satisfaction", f"{filtered_df['satisfaction_level'].mean():.2f}")
    col3.metric("Top Risk Dept.", filtered_df.groupby('department')['left'].mean().idxmax())


    show_eda(df, filtered_df)
    predict_form(df, pipeline)


if __name__ == "__main__":
    main()
