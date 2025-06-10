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
.
├── app.py                  # Streamlit app entry point
├── pipeline.py             # Pipeline + tuning logic
├── predict.py              # Churn prediction form logic
├── eda.py                  # EDA visualizations
├── utils.py                # Load data and model
├── models/                 # Serialized model pipeline
├── data/                   # Processed dataset
├── notebooks/              # Jupyter notebooks (EDA + modeling)
└── README.md
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
- High recall (91.2%) indicates strong identification of at-risk employees.

## Model Metrics (Final Random Forest)

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 96.5% |
| Precision | 88.0% |
| Recall    | 91.2% |
| F1 Score  | 89.6% |
| ROC-AUC   | 0.94  |

## Purpose

This project empowers HR teams with predictive analytics to retain top talent and reduce unexpected attrition through data-driven intervention.
