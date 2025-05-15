import streamlit as st
import pandas as pd

def predict_form(df, model):
    
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
                
