# Employee Churn Prediction with Machine Learning

## **Understand the business scenario and problem**

#### Project Goal:

- The HR department at ABC Company aims to enhance employee retention and satisfaction. They’ve collected data on various employee factors, but now they need insights into how to make better decisions based on this data.
- The goal of this project is to develop a predictive model that can determine the likelihood of an employee leaving the company.By analyzing factors , the model will provide actionable insights to help HR take proactive steps to improve employee retention and address potential issues before they lead to attrition.

This project includes EDA, feature engineering, data leakage prevention, pipeline construction, model tuning, and deployment using Streamlit.

## Project Highlights

- **Model:** Tuned Random Forest Classifier
- **Performance:** AUC = 0.94 | Accuracy = 96.5% | F1 Score = 89.6%
- **EDA:** Identified key attrition drivers — project overload, overwork, low satisfaction
- **Feature Engineering:** Created `overworked` flag; removed leakage-prone features
- **ML Pipeline:** Used `Pipeline`, `ColumnTransformer`, and `GridSearchCV`
- **Deployment:** Interactive Streamlit app for real-time risk prediction and analysis

## Folder Structure

```
─ data/                   # Processed dataset
─ models/                 # Serialized model pipeline
─ notebooks/              # Jupyter notebooks (EDA + modeling)
─ src/apps/app.py                  # Streamlit app entry point
─ src/pipeline.py             # Pipeline + tuning logic
─ src/apps/predict.py              # Churn prediction form logic
─ src/apps/eda.py                  # EDA visualizations
─ src/apps/utils.py                # Load data and model
─ README.md
```

## Setup Instructions

1. **Clone this repo:**

```bash
git clone https://github.com/your-username/employee-churn-prediction.git
cd employee-churn-prediction
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app:**

```bash
streamlit run app.py
```

---

## Key Insights

- All employees with 7 projects or >250 monthly hours had left — signs of overload.
- Feature `satisfaction_level` was dropped to prevent data leakage.
- Introduced `overworked` as a production-safe binary feature.
- High recall (90.36%) indicates strong identification of at-risk employees.

## Model Metrics for Best Selected Model(Random Forest)

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 96.2%  |
| Precision | 87%    |
| Recall    | 90.36% |
| F1 Score  | 88.67% |
| ROC-AUC   | 93.84% |

**Best Model:** The Random Forest model demonstrated the strongest performance.

**Key Metrics:**

- AUC: 93.8%  - This indicates an excellent ability to distinguish between employees who will leave and those who will stay.
- Precision: 87.0%  - This shows when the model predicts an employee will leave, it's correct around 87% (or more) of the time.
- Recall: 90.36%  - The model successfully identifies about 90% (or more) of employees who actually end up leaving.
- F1-Score: 88.7%  - This shows a strong balance between precision and recall.
- Accuracy: 96.2% .Overall, the model makes correct predictions (leave/stay) for a very high percentage of employees.

## Purpose

This project empowers HR teams with predictive analytics to retain top talent and reduce unexpected attrition through data-driven intervention.
