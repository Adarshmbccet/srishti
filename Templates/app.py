import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

# --------------------------
# Custom CSS
# --------------------------
st.markdown("""
<style>
.main-title{
    text-align:center;
    color:#1E88E5;
    font-size:40px;
    font-weight:bold;
}

.result-box{
    padding:20px;
    border-radius:10px;
    background-color:#f2f2f2;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Load Files
# --------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/logistic_model.joblib")

@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.joblib")

@st.cache_data
def load_model_info():
    with open("models/model_info.json") as f:
        return json.load(f)

@st.cache_data
def load_feature_ranges():
    with open("models/feature_ranges.json") as f:
        return json.load(f)

model = load_model()
scaler = load_scaler()
info = load_model_info()
ranges = load_feature_ranges()

# --------------------------
# Sidebar
# --------------------------

st.sidebar.title("📊 Model Information")

st.sidebar.write("### Model")
st.sidebar.success(info["model_type"])

st.sidebar.write("### Accuracy")
st.sidebar.info(f"{info['accuracy']*100:.2f}%")

st.sidebar.write("### Features")

for feature in info["feature_names"]:
    st.sidebar.write("•", feature)

# --------------------------
# Title
# --------------------------

st.markdown(
    "<div class='main-title'>🛒 Customer Purchase Prediction</div>",
    unsafe_allow_html=True
)

st.write(
    "Predict whether a customer will purchase a product based on **Age** and **Estimated Salary**."
)

# --------------------------
# Input Section
# --------------------------

st.header("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        min_value=int(ranges["Age"]["min"]),
        max_value=int(ranges["Age"]["max"]),
        value=int(ranges["Age"]["default"])
    )

with col2:

    salary = st.slider(
        "Estimated Salary",
        min_value=int(ranges["EstimatedSalary"]["min"]),
        max_value=int(ranges["EstimatedSalary"]["max"]),
        value=int(ranges["EstimatedSalary"]["default"]),
        step=1000
    )

# --------------------------
# Display Current Input
# --------------------------

st.subheader("Current Input")

df = pd.DataFrame({
    "Feature":["Age","Estimated Salary"],
    "Value":[age,salary]
})

st.dataframe(df, use_container_width=True, hide_index=True)

# --------------------------
# Prediction
# --------------------------

if st.button("Predict Purchase", use_container_width=True):

    input_data = np.array([[age,salary]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction==1:

        st.success("✅ Customer WILL Purchase")

    else:

        st.error("❌ Customer will NOT Purchase")

    st.subheader("Prediction Confidence")

    st.write("Not Purchased")

    st.progress(float(probability[0]))

    st.write(f"{probability[0]*100:.2f}%")

    st.write("Purchased")

    st.progress(float(probability[1]))

    st.write(f"{probability[1]*100:.2f}%")

# --------------------------
# About Dataset
# --------------------------

with st.expander("📚 About Dataset"):

    st.write("""
The **Social Network Ads** dataset contains customer information.

### Features

- Age
- Estimated Salary

### Target

- 0 → Not Purchased
- 1 → Purchased

The Logistic Regression model predicts whether a customer is likely to purchase a product based on these features.
""")

# --------------------------
# Footer
# --------------------------

st.markdown("---")

st.markdown(
    "<center>Built with ❤️ using Streamlit & Scikit-learn</center>",
    unsafe_allow_html=True
)