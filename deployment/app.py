import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json
from PIL import Image
import io
# Set page config
st.set_page_config(
page_title="Iris Flower Classifier",
page_icon="🌸",
layout="wide",
initial_sidebar_state="expanded"
)
# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #6a0dad;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #6a0dad;
        margin: 1rem 0;
    }
    .feature-slider {
        margin: 1.5rem 0;
    }
    .confidence-bar {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        text-align: center;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)
# Load model and metadata
@st.cache_resource
def load_model(format_type='joblib'):
    """Load the model from the specified format"""
    try:
       if format_type == 'joblib':
          model = joblib.load('models/iris_model.joblib')
       elif format_type == 'pickle':
          with open('models/iris_model.pickle', 'rb') as f:
            model = pickle.load(f)
       return model
    except Exception as e:
    
     st.error(f"Error loading model: {e}")
     return None
@st.cache_resource
def load_model_info():
  """Load model metadata"""
  try:
   with open('models/model_info.json', 'r') as f:
    return json.load(f)
  except Exception as e:
   st.error(f"Error loading model info: {e}")
  return None
@st.cache_resource
def load_feature_ranges():
 """Load feature ranges for sliders"""
 try:
   with open('models/feature_ranges.json', 'r') as f:
    return json.load(f)
 except:
# Default ranges if file doesn't exist
  return {
'sepal_length': {'min': 4.0, 'max': 8.0, 'default': 5.8},
'sepal_width': {'min': 2.0, 'max': 4.5, 'default': 3.0},
'petal_length': {'min': 1.0, 'max': 7.0, 'default': 4.0},
'petal_width': {'min': 0.1, 'max': 2.5, 'default': 1.2}
}
# Load data
model_info = load_model_info()
feature_ranges = load_feature_ranges()
model = load_model('joblib')  # Default to joblib
# Sidebar
with st.sidebar:
 st.title("⚙Settings")
# Model format selection
model_format = st.radio(
"Model Format",
["joblib", "pickle"],
help="Choose which serialized model format to use for predictions"
)
# Reload model button
if st.button("🔄Reload Model"):
 model = load_model(model_format)
if model:
 st.success(f"Model loaded from {model_format} format!")
st.divider()
# Model info
st.subheader("📊Model Information")
if model_info:
 st.write(f"**Type:** {model_info.get('model_type', 'RandomForest')}")
st.write(f"**Accuracy:** {model_info.get('accuracy', 0.96):.1%}")
st.write(f"**Features:** {len(model_info.get('feature_names', []))}")
st.write(f"**Classes:** {len(model_info.get('target_names', []))}")
st.divider()
# Quick actions
st.subheader("🚀Quick Actions")
if st.button("📊Show Dataset Info"):
 st.session_state.show_dataset_info = True
if st.button("🎯Make Prediction"):
 st.session_state.make_prediction = True
# Main content
st.markdown('<h1 class="main-header">🌸Iris Flower Classification</h1>', unsafe_allow_html=True)
# Description
st.markdown("""
This app predicts the species of an Iris flower based on its measurements using a machine lear
Adjust the sliders below to input the flower's characteristics and see the prediction!
""")
# Create input form
col1, col2 = st.columns([2, 1])
with col1:
  st.header("📝Input Features")
# Create sliders based on feature ranges
sepal_length = st.slider(
"**Sepal Length (cm)**",
min_value=float(feature_ranges['sepal_length']['min']),
max_value=float(feature_ranges['sepal_length']['max']),
value=float(feature_ranges['sepal_length']['default']),
step=0.1,
help="Length of the sepal in centimeters"
)
sepal_width = st.slider(
"**Sepal Width (cm)**",
min_value=float(feature_ranges['sepal_width']['min']),
max_value=float(feature_ranges['sepal_width']['max']),
value=float(feature_ranges['sepal_width']['default']),
step=0.1,
help="Width of the sepal in centimeters"
)
petal_length = st.slider(
"**Petal Length (cm)**",
min_value=float(feature_ranges['petal_length']['min']),
max_value=float(feature_ranges['petal_length']['max']),
value=float(feature_ranges['petal_length']['default']),
step=0.1,
help="Length of the petal in centimeters"
)
petal_width = st.slider(
"**Petal Width (cm)**",
min_value=float(feature_ranges['petal_width']['min']),
max_value=float(feature_ranges['petal_width']['max']),
value=float(feature_ranges['petal_width']['default']),
step=0.1,
help="Width of the petal in centimeters"
)
with col2:
 st.header("📊Current Values")
# Display current feature value
# Display current feature values
features_df = pd.DataFrame({
'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
})
st.dataframe(
features_df,
hide_index=True,
use_container_width=True
)
# Create feature array for prediction
input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
# Prediction button
if st.button("🎯Predict Species", type="primary", use_container_width=True):
  if model is not None and model_info is not None:
    try:
      prediction = model.predict(input_features)
      prediction_proba = model.predict_proba(input_features)[0]
# Get predicted class name
      predicted_class = model_info['target_names'][prediction[0]]
# Display results
      st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
      st.markdown(f"### ��Prediction Result")
      st.markdown(f"**Predicted Species:** **{predicted_class}**")
# Show confidence scores with progress bars
      st.markdown("### ��Confidence Scores")
      for i, prob in enumerate(prediction_proba):
       species = model_info['target_names'][i]
       percentage = prob * 100
       col_prog, col_text = st.columns([3, 1])
       with col_prog:
         st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {percentage}%;">
                         {percentage:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
      with col_text:
         st.write(f"**{species}**")
      st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
     st.error(f"❌Error making prediction: {e}")
  else:
   st.error("❌Model could not be loaded. Please check if the model files exist.")
# Additional information
with st.expander("📚About the Iris Dataset"):
  st.markdown("""The Iris flower dataset is a classic dataset in machine learning and statistics introduce
  **Dataset Characteristics:**
     - 150 samples (50 per class)
    - 4 features per sample
    - 3 classes (species)
    **Species:**
    - **Iris Setosa**
    - **Iris Versicolor** 
    - **Iris Virginica**
    **Features:**
    1. Sepal length (cm)
    2. Sepal width (cm)
    3. Petal length (cm)
    4. Petal width (cm)
    This model uses a **Random Forest classifier** with an accuracy of approximately 96%.
    """)
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit and Scikit-learn</p>
</div>
""", unsafe_allow_html=True)