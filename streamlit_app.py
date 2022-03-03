from joblib import load
from sklearn import tree, datasets
import streamlit as st

iris = datasets.load_iris()

clf = load("iris.joblib")
st.title("SMU Model Demo")

labels = iris.target_names
inputs = iris.feature_names

var_1 = st.slider(inputs[0], max_value=10)
var_2 = st.slider(inputs[1], max_value=10)
var_3 = st.slider(inputs[2], max_value=10)
var_4 = st.slider(inputs[3], max_value=10)

prediction = clf.predict([[var_1, var_2, var_3, var_4]])

st.write(prediction[0])
