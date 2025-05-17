
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_eda(df, filtered_df):
    
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
        ax[0].set_ylabel("Tenure (Years)")



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
