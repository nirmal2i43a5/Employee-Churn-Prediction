import streamlit as st
import pandas as pd
import pickle


# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv('data/preprocessed_dataset.csv')


@st.cache_resource
def load_model():
    with open('models/hr_rf1.pickle', 'rb') as f_model:
        model = pickle.load(f_model)
  
        return model
