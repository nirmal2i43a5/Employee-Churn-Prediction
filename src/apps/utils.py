import streamlit as st
import pandas as pd
import joblib
import pickle


# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv('data/preprocessed_dataset.csv')


@st.cache_resource
def load_pipeline():
    with open('models/rf2_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline


@st.cache_resource
def load_model():
    model = joblib.load('models/hr_rf2.pickle')
    return model
