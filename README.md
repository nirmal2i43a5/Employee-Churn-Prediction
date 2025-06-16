Employee Churn Prediction with Machine Learning

- This project uses machine learning to predict whether an employee will leave the company based on HR records.
- It includes complete data exploration, preprocessing, feature engineering, model selection, evaluation, and deployment using a Streamlit web app.


## Demo

Or try the live app 👉 [Streamlit Cloud Link](https://employee-churn-prediction.streamlit.app)

## **Understand the business scenario and problem**

#### Project Goal:

- Help HR to identify employees at high risk of leaving the company.
- Provide actionable, data-driven insights to improve retention.
- Build a **reliable**, **interpretable**, and **production-ready** ML system using scikit-learn pipelines and Streamlit.

## Key EDA Insights

- **Overwork/projects seems to a major attrition signal**: Employees with 6+ projects or >250 monthly hours had high leave rates.
- **Satisfaction levels and tenure** show a strong relationship with leaving.
- **Promotion history** and **salary level** showed weaker, but still relevant, patterns.
- A new binary feature **`overworked`** was introduced.

## Modeling & Pipeline

### Models Used:

- Logistic Regression (baseline Model)
- Decision Tree Classifier
- Random Forest Classifier (**best performer**)

### Pipeline Design:

- Used `Pipeline` and `ColumnTransformer` for clean, modular ML design.
- Tuned using `GridSearchCV` with multiple metrics:
  `accuracy`, `precision`, `recall`, `f1`, and `roc_auc`.

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
streamlit run src/apps/app.py
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


## Project Highlights

- **Model:** Tuned Random Forest Classifier
- **Performance:** AUC = 93.8% | Accuracy = 96.2% | F1 Score = 88.67% | Recall = 90.36 %
- **EDA:** Identified key attrition drivers — project overload, overwork, low satisfaction
- **Feature Engineering:** Created `overworked` flag; removed leakage-prone features
- **ML Pipeline:** Used `Pipeline`, `ColumnTransformer`, and `GridSearchCV`
- **Deployment:** Interactive Streamlit app for real-time risk prediction and analysis
