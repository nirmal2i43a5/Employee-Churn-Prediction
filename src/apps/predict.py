import streamlit as st
import pandas as pd

def predict_form(df, model):#model pipeline 
    
    st.subheader("Predict Attrition")
    
    with st.form("predict_form"):
        
        # satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
        last_eval = st.slider("Performance Review Score", 0.0, 1.0, 0.5)
        projects = st.number_input("Number of Projects", 1, 10, 3)
        # average_monthly_hours = st.slider("Average Monthly Hours", 80, 320, 160)
        overworked = st.checkbox(
        "Overworked (Hours > 175)", value=False
    )
        tenure = st.slider("Tenure(In Years)", 1, 10, 3)
        department = st.selectbox("Department", df['department'].unique())
        salary = st.selectbox("Salary Level", df['salary'].unique())

        submitted = st.form_submit_button("Predict")
        if submitted:
            input_data = {
                # 'satisfaction_level': satisfaction,
                'last_evaluation': last_eval,
                'number_project': projects,
                'overworked': overworked,
                'tenure': tenure,
                'work_accident': 0,
                'promotion_last_5years': 0,
                'department': department,
                'salary': salary
            }
            input_df = pd.DataFrame([input_data])

   
            prediction_probability = model.predict_proba(input_df)[0][1]
            prediction = model.predict(input_df)[0]
            
            st.success(f"Probability of Leaving: {prediction_probability:.2%}")

            if prediction_probability >= 0.7:
                st.error("High Risk: This employee is very likely to leave.")
            elif 0.3 <= prediction_probability < 0.7:
                st.warning("Medium Risk: This employee might leave. Monitor closely.")
            else:
                st.success("Low Risk: This employee is likely to stay.")
                
