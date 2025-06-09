import streamlit as st
import pandas as pd
import pickle
from sklearn.svm import SVR

# Load trained model
@st.cache_resource
def load_model():
    with open('svm_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# App Title
st.title("🎬 Movie Rating Predictor")
st.markdown("Enter the vote count for a movie to predict its average rating.")

# Input from user
vote_count = st.number_input("Enter vote count:", min_value=1, step=10)

# Prediction
if st.button("Predict Vote Average"):
    prediction = model.predict([[vote_count]])[0]
    st.success(f"📈 Predicted Vote Average: **{prediction:.2f}**")
