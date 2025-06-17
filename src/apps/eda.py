
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def show_eda(df, filtered_df):
    st.subheader("Employee Churn Analysis Dashboard")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Stayed/Left by Dept", 
        "Satisfaction Trend", 
        "Promotion Impact", 
        "Heatmap",
        "Satisfaction and Tenure Analysis"
    ])
    
    with tab1:
        # Creating a count of employees by department and left status
        fig = px.histogram(
            filtered_df,
            x='department',
            color='left',
            barmode='group',
            color_discrete_sequence=['#3498db', '#e74c3c'],
            labels={'left': 'Status', 'department': 'Department'},
            category_orders={"left": [0, 1]},
            title='Counts of Stayed vs Left by Department'
        )
        fig.update_layout(
            xaxis_title="Department",
            yaxis_title="Count",
            legend_title="Status",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
      
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        avg_satisfaction = filtered_df.groupby('tenure')['satisfaction_level'].mean().reset_index()
        fig = px.line(
            avg_satisfaction, 
            x='tenure', 
            y='satisfaction_level',
            markers=True,
            title='Satisfaction Trend by Tenure',
            labels={'satisfaction_level': 'Average Satisfaction', 'tenure': 'Years at Company'}
        )
        fig.update_layout(
            xaxis_title="Tenure (Years)",
            yaxis_title="Average Satisfaction",
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Promotion Impact on Attrition")
        
        # Count employees by promotion and left status
        promotion_counts = filtered_df.groupby(['promotion_last_5years', 'left']).size().reset_index(name='count')
        
        # Calculate percentages within each promotion group
        total_by_promotion = promotion_counts.groupby('promotion_last_5years')['count'].transform('sum')
        promotion_counts['percentage'] = promotion_counts['count'] / total_by_promotion * 100
        
        fig = px.bar(
            promotion_counts,
            x='promotion_last_5years',
            y='percentage',
            color='left',
            barmode='stack',
            color_discrete_sequence=['#3498db', '#e74c3c'],
            labels={'promotion_last_5years': 'Promoted in Last 5 Years', 'percentage': 'Percentage (%)'},
            title='Attrition Rate by Promotion in Last 5 Years',
            text=promotion_counts['percentage'].round(1).astype(str) + '%'
        )
        
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=[0, 1],
                ticktext=['No Promotion', 'Promoted']
            ),
            yaxis_title="Percentage (%)",
            legend_title="Status"
        )
        # Update legend text
        if len(fig.data) > 0:
            fig.data[0].name = 'Stayed'
        if len(fig.data) > 1:
            fig.data[1].name = 'Left'
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        corr = filtered_df.corr(numeric_only=True)
        
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Feature Correlation Heatmap"
        )
        fig.update_layout(
            height=600,
            width=700
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        
        fig1 = px.box(
            filtered_df,
            x='tenure',
            y='satisfaction_level',
            color='left',
            color_discrete_sequence=['#3498db', '#e74c3c'],
            title='Satisfaction by Tenure',
            labels={'satisfaction_level': 'Satisfaction Level', 'tenure': 'Tenure (Years)'}
        )
        fig1.update_layout(
            xaxis_title="Tenure (Years)",
            yaxis_title="Satisfaction Level",
            legend_title="Status"
        )
      
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.histogram(
            filtered_df,
            x='tenure',
            color='left',
            barmode='group',
            title='Tenure Histogram (Stayed vs Left)',
            color_discrete_sequence=['#3498db', '#e74c3c'],
            category_orders={"left": [0, 1]}
        )
        fig2.update_layout(
            xaxis_title="Years at Company",
            yaxis_title="Count",
            legend_title="Status"
        )
      
        st.plotly_chart(fig2, use_container_width=True)




    # High Risk Employee List
    st.subheader("High-Risk Employee List")
    risk_df = df[(df['satisfaction_level'] < 0.4) & (df['average_monthly_hours'] > 250)]
    
    # Add a risk score column (optional enhancement)
    risk_df['risk_score'] = (1 - risk_df['satisfaction_level']) * (risk_df['average_monthly_hours'] / 300) * 10
    risk_df['risk_score'] = risk_df['risk_score'].round(1)
    
    # Display the dataframe with the new column
    st.dataframe(
        risk_df[['department', 'number_project', 'tenure', 'satisfaction_level', 
                'salary', 'average_monthly_hours', 'risk_score', 'left']]
    )
    
    # Download button for the high-risk employee data
    st.download_button(
        "Download High-Risk Data", 
        risk_df.to_csv(index=False), 
        "high_risk_employees.csv"
    )