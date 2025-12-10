import joblib
import numpy as np
import streamlit as st

model = joblib.load("../models/iris_model.joblib")

st.title("🌸 Iris Flower Prediction App")
st.write("Enter the features below to predict the Iris species.")

sepal_length = st.number_input("Sepal Length", 0.0, 10.0, 5.1)
sepal_width  = st.number_input("Sepal Width",  0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length", 0.0, 10.0, 1.4)
petal_width  = st.number_input("Petal Width",  0.0, 10.0, 0.2)

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(features)[0]

    species_map = {0: "Setosa", 1: "Versicolour", 2: "Virginica"}
    st.success(f"Predicted Species: **{species_map[pred]}**")
