import streamlit as st
# Title & text
st.title("Hello Streamlit ")
st.write("This is your first Streamlit app!")
# Input from user
name = st.text_input("Enter your name:")
if name:
    st.success(f"Hello {name}, welcome to Streamlit!")
st.title("Main Title")
st.header("Header")
st.subheader("Subheader")
st.text("Simple text")
st.markdown("**Bold text** with *italics* and `code`")
# Button
if st.button("Click Me"):
    st.write("Button clicked!")
# Checkbox
agree = st.checkbox("I agree")
if agree:
    st.write("✅You agreed!")
# Radio buttons
choice = st.radio("Choose one:", ["Option 1", "Option 2"])
st.write("You selected:", choice)
# Selectbox
option = st.selectbox("Pick a number:", [1, 2, 3, 4, 5])
st.write("Your number is:", option)
# Slider
value = st.slider("Select a range", 0, 100, 25)
st.write("Slider value:", value)
import pandas as pd
import numpy as np
df = pd.DataFrame(
    np.random.randn(10, 3),
    columns=["A", "B", "C"]
)
st.write("Random DataFrame:", df)
st.line_chart(df)
st.bar_chart(df)
# Sidebar
st.sidebar.title("Sidebar Menu")
user = st.sidebar.text_input("Enter your username")
st.sidebar.write("Welcome", user)
# Columns
col1, col2 = st.columns(2)
col1.write("This is column 1")
col2.write("This is column 2")
uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())