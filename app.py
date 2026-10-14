import streamlit as st
import pickle
from sklearn.datasets import load_iris

st.title("🌸 ML Prediction App")
st.write("Iris Flower Prediction")

data = load_iris()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

a = st.number_input("Sepal length", 0.0, 10.0, 5.1)
b = st.number_input("Sepal width", 0.0, 10.0, 3.5)
c = st.number_input("Petal length", 0.0, 10.0, 1.4)
d = st.number_input("Petal width", 0.0, 10.0, 0.2)

if st.button("Predict"):
    result = model.predict([[a, b, c, d]])[0]
    st.success(f"Prediction: {data.target_names[result]}")