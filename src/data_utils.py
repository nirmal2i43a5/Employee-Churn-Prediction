
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_PATH = BASE_DIR / "models"

def load_raw_data():
   
    print("Loading raw data...")
    df0 = pd.read_csv(DATA_PATH)
    print(f"Raw data shape: {df0.shape}")
    return df0


def clean_data(df0):
  
    print("Cleaning data...")
    
    # Rename columns to standardize naming
    df0 = df0.rename(columns={
        'Work_accident': 'work_accident',
        'average_montly_hours': 'average_monthly_hours',
        'time_spend_company': 'tenure',
        'Department': 'department'
    })
    
    # Remove duplicates
    df1 = df0.drop_duplicates(keep='first')
    
    print(f"Data shape after cleaning: {df1.shape}")
    print(f"Removed {df0.shape[0] - df1.shape[0]} duplicate rows")
    
    return df1


def create_overworked_feature(df1):

    print("Creating overworked feature...")
    
    df2 = df1.copy()
    
    # Drop satisfaction_level to avoid data leakage
    df2 = df2.drop('satisfaction_level', axis=1)
    
    # Create overworked feature
    df2['overworked'] = (df2['average_monthly_hours'] > 175).astype(int)
    
    # Drop average_monthly_hours as we now have the overworked feature
    df2 = df2.drop('average_monthly_hours', axis=1)
    
    print(f"Data shape after feature engineering: {df2.shape}")
    return df2


def encode_categorical_variables(df2):
  
    print("Encoding categorical variables...")
    
    df_enc = df2.copy()
    
    # Encode salary as ordinal (low=0, medium=1, high=2)
    df_enc['salary'] = (
        df_enc['salary'].astype('category')
        .cat.set_categories(['low', 'medium', 'high'])
        .cat.codes
    )
    
    # Dummy encode department
    df_enc = pd.get_dummies(df_enc, drop_first=False)
    
    print(f"Encoded data shape: {df_enc.shape}")
    return df_enc


def prepare_modeling_data(df_enc, remove_outliers=False):
  
    print("Preparing data for modeling...")
    
    if remove_outliers:
        # Calculate outlier limits for tenure
        percentile25 = df_enc['tenure'].quantile(0.25)
        percentile75 = df_enc['tenure'].quantile(0.75)
        iqr = percentile75 - percentile25
        upper_limit = percentile75 + 1.5 * iqr
        lower_limit = percentile25 - 1.5 * iqr
        
        # Remove outliers
        df_model = df_enc[(df_enc['tenure'] >= lower_limit) & 
                          (df_enc['tenure'] <= upper_limit)]
        print(f"Data shape after removing outliers: {df_model.shape}")
    else:
        df_model = df_enc
    
    # Separate features and target
    y = df_model['left']
    X = df_model.drop('left', axis=1)
    
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def get_data_summary(df):
  
    print("\n=== Data Summary ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    if 'left' in df.columns:
        print(f"Target distribution:")
        print(df['left'].value_counts(normalize=True))


# Convenience functions
def load_cleaned_data():
    df0 = load_raw_data()
    df1 = clean_data(df0)
    return df1


def load_encoded_data():
    df0 = load_raw_data()
    df1 = clean_data(df0)
    df2 = create_overworked_feature(df1)
    df_enc = encode_categorical_variables(df2)
    return df_enc


def load_modeling_data(remove_outliers=False):
    df_enc = load_encoded_data()
    X, y = prepare_modeling_data(df_enc, remove_outliers)
    return X, y
