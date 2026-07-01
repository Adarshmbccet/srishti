import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Title
st.title("🌸 Iris Flower Prediction App")
st.write("Enter flower measurements to predict the Iris species.")

# Load Dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Sidebar Inputs
st.sidebar.header("Flower Measurements")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Input DataFrame
input_data = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=iris.feature_names
)

st.subheader("📋 Input Values")
st.dataframe(input_data)

# Prediction
prediction = clf.predict(input_data)
prediction_proba = clf.predict_proba(input_data)

species = iris.target_names[prediction[0]]
confidence = max(prediction_proba[0]) * 100

# Output
st.subheader("🌺 Prediction Result")
st.success(f"Predicted Species: **{species.upper()}**")

st.subheader("📊 Confidence")
st.progress(int(confidence))
st.write(f"Confidence: **{confidence:.2f}%**")

# Probability Table
st.subheader("🔍 Species Probabilities")
prob_df = pd.DataFrame(
    prediction_proba,
    columns=iris.target_names
)
st.dataframe(prob_df.style.format("{:.2%}"))