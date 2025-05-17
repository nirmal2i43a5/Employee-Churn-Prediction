import streamlit as st
import pandas as pd
import joblib


# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv('data/preprocessed_dataset.csv')


@st.cache_resource
def load_pipeline():
    pipeline = joblib.load('models/rf2_pipeline.pkl')
    return pipeline
