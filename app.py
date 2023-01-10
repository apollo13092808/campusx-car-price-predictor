import pickle
import numpy as np
import pandas as pd
import streamlit as st


model = pickle.load(open("pickles/model.pkl", "rb"))
df = pickle.load(open("pickles/data.pkl", "rb"))


st.header("Car Price Predictor")

selected_company = st.selectbox(
    label="Select company here...",
    options=sorted(df["company"].unique())
)

selected_model = st.selectbox(
    label="Select model here...",
    options=sorted(df["name"].unique())
)

selected_year = st.selectbox(
    label="Select manufactured year here...",
    options=sorted(df["year"].unique(), reverse=True)
)

selected_fueltype = st.selectbox(
    label="Select fuel type here...",
    options=sorted(df["fuel_type"].unique())
)

distance = st.number_input(label="Kilometers driven...", min_value=0)

if st.button(label="Predict"):
    predicted = model.predict(pd.DataFrame(
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
        data=np.array([selected_model, selected_company, selected_year, distance, selected_fueltype]).reshape(1, 5))
    )

    result = np.round(predicted[0], 3)
    st.header(f"Predicted Price is {result}")
