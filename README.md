# Employee Churn Prediction with Machine Learning

## **Understand the business scenario and problem**

#### Project Goal:

- The HR department at ABC Company aims to enhance employee retention and satisfaction. They’ve collected data on various employee factors, but now they need insights into how to make better decisions based on this data.
- The goal of this project is to develop a predictive model that can determine the likelihood of an employee leaving the company.By analyzing factors , the model will provide actionable insights to help HR take proactive steps to improve employee retention and address potential issues before they lead to attrition.

This project includes EDA, feature engineering, data leakage prevention, pipeline construction, model tuning, and deployment using Streamlit.

## Project Highlights

- **Model:** Tuned Random Forest Classifier
- **Performance:** AUC = 93.8% | Accuracy = 96.2% | F1 Score = 88.67% | Recall = 90.36 %
- **EDA:** Identified key attrition drivers — project overload, overwork, low satisfaction
- **Feature Engineering:** Created `overworked` flag; removed leakage-prone features
- **ML Pipeline:** Used `Pipeline`, `ColumnTransformer`, and `GridSearchCV`
- **Deployment:** Interactive Streamlit app for real-time risk prediction and analysis

## Folder Structure

```
─ data/                   # Processed dataset
─ models/                 # Serialized model pipeline
─ notebooks/              # Jupyter notebooks (EDA + modeling)
─ src/apps/app.py         # Streamlit app entry point
─ src/pipeline.py         # Pipeline + tuning logic
─ src/apps/predict.py     # Churn prediction form logic
─ src/apps/eda.py         # EDA visualizations
─ src/apps/utils.py       # Load data and model
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
- Recall: 90.36%  - The model successfully identifies about 90% (or more) of employees who actually end up leaving and also shows strong identification of at-risk employees.
- F1-Score: 88.7%  - This shows a strong balance between precision and recall.
- Accuracy: 96.2% .Overall, the model makes correct predictions (leave/stay) for a very high percentage of employees.

## Summary of model results

**Logistic Regression**

The logistic regression model achieved precision of 80%, recall of 83%, f1-score of 80% (all weighted averages), and accuracy of 83%, on the test set.

**Tree-based Machine Learning**

After conducting feature engineering, the decision tree model achieved AUC of 93.8%, precision of 87.0%, recall of 90.4%, f1-score of 88.7%, and accuracy of 96.2%, on the test set. The random forest modestly outperformed the decision tree model.

## Purpose

This project empowers HR teams with predictive analytics to retain top talent and reduce unexpected attrition through data-driven intervention.
