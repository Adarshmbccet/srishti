import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>

/* Main Background */
.stApp{
    background: linear-gradient(135deg,#0F2027,#203A43,#2C5364);
}

/* Remove Streamlit Menu */
#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

/* Hero Card */

.hero{
background:linear-gradient(135deg,#005AA7,#00CDAC);
padding:30px;
border-radius:20px;
box-shadow:0px 10px 25px rgba(0,0,0,.35);
text-align:center;
margin-bottom:25px;
}

.hero h1{
color:white;
font-size:48px;
margin-bottom:10px;
}

.hero p{
color:white;
font-size:20px;
}

/* Glass Card */

.glass{

background:rgba(255,255,255,.12);

backdrop-filter:blur(12px);

padding:20px;

border-radius:18px;

box-shadow:0 8px 30px rgba(0,0,0,.25);

margin-bottom:20px;

}

/* Metric Card */

.metric-card{

background:white;

padding:20px;

border-radius:15px;

text-align:center;

box-shadow:0px 4px 20px rgba(0,0,0,.18);

transition:.3s;

}

.metric-card:hover{

transform:translateY(-6px);

}

/* Prediction */

.success-card{

background:#d4edda;

padding:20px;

border-radius:15px;

border-left:8px solid green;

}

.fail-card{

background:#f8d7da;

padding:20px;

border-radius:15px;

border-left:8px solid red;

}

/* Sidebar */

section[data-testid="stSidebar"]{

background:#132743;

}

/* Buttons */

.stButton>button{

background:linear-gradient(90deg,#00b09b,#96c93d);

color:white;

font-size:20px;

font-weight:bold;

border:none;

padding:15px;

border-radius:12px;

width:100%;

transition:.3s;

}

.stButton>button:hover{

transform:scale(1.03);

box-shadow:0px 8px 25px rgba(0,0,0,.3);

}

</style>
""",unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/titanic_model.joblib")

@st.cache_data
def load_model_info():
    with open("models/model_info.json") as f:
        return json.load(f)

@st.cache_data
def load_ranges():
    with open("models/feature_ranges.json") as f:
        return json.load(f)

model=load_model()
info=load_model_info()
ranges=load_ranges()

# ----------------------------
# HERO SECTION
# ----------------------------

st.markdown("""
<div class="hero">

<h1>🚢 Titanic Survival Prediction</h1>

<p>
Predict passenger survival using an AI-powered Random Forest model.
</p>

</div>
""",unsafe_allow_html=True)

# ----------------------------
# DASHBOARD METRICS
# ----------------------------

m1,m2,m3,m4=st.columns(4)

with m1:

    st.markdown(f"""
<div class="metric-card">

<h3 style="color:#555;">🤖 Model</h3>

<h2 style="color:#0d6efd;">
    {info['model_type']}
</h2>

</div>
""", unsafe_allow_html=True)

    


with m2:

    st.markdown(f"""
<div class="metric-card">

<h3 style="color:#555;">🎯 Accuracy</h3>

<h2 style="color:#28a745;">
{info['accuracy']*100:.2f}%
</h2>

</div>
""", unsafe_allow_html=True)
with m3:

    st.markdown("""
<div class="metric-card">

<h3 style="color:#555;">📂 Dataset</h3>

<h2 style="color:#fd7e14;">
891
</h2>

</div>
""", unsafe_allow_html=True)
with m4:

    st.markdown(f"""
<div class="metric-card">

<h3 style="color:#555;">📊 Features</h3>

<h2 style="color:#6f42c1;">
{len(info['feature_names'])}
</h2>

</div>
""", unsafe_allow_html=True)
# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg",
        use_container_width=True
    )

    st.markdown("## 🚢 Model Dashboard")

    st.success("✅ Model Loaded")

    st.metric(
        label="Accuracy",
        value=f"{info['accuracy']*100:.2f}%"
    )

    st.metric(
        label="Algorithm",
        value=info["model_type"]
    )

    st.metric(
        label="Passengers",
        value="891"
    )

    st.metric(
        label="Features",
        value=len(info["feature_names"])
    )

    st.divider()

    st.info(
        """
This dashboard predicts whether a passenger
would likely survive based on historical
Titanic passenger data.
"""
    )


# ==========================================================
# INPUT SECTION
# ==========================================================

st.markdown(
"""
<div class='glass'>

<h2 style="color:white;">👤 Passenger Information</h2>

<p style="color:white;">
Enter passenger details below.
</p>

</div>
""",
unsafe_allow_html=True
)

left,right=st.columns([2,1])

# ==========================================================
# LEFT PANEL
# ==========================================================

with left:

    pclass=st.selectbox(

        "🎟 Passenger Class",

        [1,2,3],

        help="Higher class passengers generally had a better survival rate."

    )

    sex=st.radio(

        "👤 Gender",

        ["Male","Female"],

        horizontal=True

    )

    age=st.slider(

        "🎂 Age",

        min_value=0,

        max_value=80,

        value=30

    )

    fare=st.slider(

        "💰 Ticket Fare",

        min_value=0.0,

        max_value=600.0,

        value=50.0,

        step=5.0

    )

    sibsp=st.slider(

        "👨‍👩‍👧 Siblings / Spouse",

        0,

        8,

        0

    )

    parch=st.slider(

        "👪 Parents / Children",

        0,

        6,

        0

    )

    embarked=st.selectbox(

        "🚢 Embarked",

        ["C","Q","S"],

        help="Port where passenger boarded."

    )

# ==========================================================
# RIGHT PANEL
# ==========================================================

with right:

    st.markdown(
    """
    <div class='glass'>

    <h3 style="color:white;">
    📋 Live Passenger Summary
    </h3>

    </div>
    """,
    unsafe_allow_html=True
    )

    gender_icon="👨" if sex=="Male" else "👩"

    summary=pd.DataFrame({

        "Feature":[

            "🎟 Class",

            "Gender",

            "🎂 Age",

            "💰 Fare",

            "👨‍👩‍👧 SibSp",

            "👪 Parch",

            "🚢 Embarked"

        ],

        "Value":[

            pclass,

            f"{gender_icon} {sex}",

            age,

            f"${fare:.2f}",

            sibsp,

            parch,

            embarked

        ]

    })

    st.dataframe(

        summary,

        hide_index=True,

        use_container_width=True

    )

    st.success("✔ Input Ready")

    st.caption(
        "Press the Predict button below."
    )
# ==========================================================
# PREPARE INPUT
# ==========================================================

sex_value = 1 if sex == "Male" else 0

embarked_dict = {
    "C": 0,
    "Q": 1,
    "S": 2
}

embarked_value = embarked_dict[embarked]

input_data = np.array([[
    pclass,
    sex_value,
    age,
    sibsp,
    parch,
    fare,
    embarked_value
]])

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# PREDICT BUTTON
# ==========================================================

predict = st.button(
    "🚀 Predict Survival",
    use_container_width=True
)

# ==========================================================
# PREDICTION
# ==========================================================

if predict:

    with st.spinner("🧠 AI Model is analysing passenger details..."):

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]

    st.balloons()

    st.markdown("---")

    st.subheader("🎯 Prediction Result")

    result_left, result_right = st.columns([2,1])

    # ======================================================
    # RESULT CARD
    # ======================================================

    with result_left:

        if prediction == 1:

            st.markdown(
            """
            <div class="success-card">

            <h2>
            ✅ Passenger is likely to SURVIVE
            </h2>

            <h4>
            Survival chances are high.
            </h4>

            </div>
            """,
            unsafe_allow_html=True
            )

        else:

            st.markdown(
            """
            <div class="fail-card">

            <h2>
            ❌ Passenger is NOT likely to survive
            </h2>

            <h4>
            Survival chances are low.
            </h4>

            </div>
            """,
            unsafe_allow_html=True
            )

    # ======================================================
    # AI CONFIDENCE CARD
    # ======================================================

    with result_right:

        confidence = np.max(probability) * 100

        st.metric(

            "🤖 AI Confidence",

            f"{confidence:.2f}%"

        )

    st.markdown("---")

    # ======================================================
    # PROBABILITY SECTION
    # ======================================================

    st.subheader("📊 Prediction Confidence")

    col1, col2 = st.columns(2)

    with col1:

        st.write("❌ Did Not Survive")

        st.progress(float(probability[0]))

        st.info(f"{probability[0]*100:.2f}%")

    with col2:

        st.write("✅ Survived")

        st.progress(float(probability[1]))

        st.success(f"{probability[1]*100:.2f}%")

    st.markdown("---")

    # ======================================================
    # PASSENGER SUMMARY CARD
    # ======================================================

    st.subheader("📋 Passenger Summary")

    summary = pd.DataFrame({

        "Feature":[

            "Passenger Class",

            "Gender",

            "Age",

            "Fare",

            "Siblings / Spouse",

            "Parents / Children",

            "Embarked"

        ],

        "Value":[

            pclass,

            sex,

            age,

            f"${fare:.2f}",

            sibsp,

            parch,

            embarked

        ]

    })

    st.dataframe(

        summary,

        hide_index=True,

        use_container_width=True

    )

    st.markdown("---")

    # ======================================================
    # AI ANALYSIS
    # ======================================================

    st.subheader("🤖 AI Analysis")

    analysis = []

    if sex == "Female":
        analysis.append("👩 Female passengers historically had a higher survival rate.")

    else:
        analysis.append("👨 Male passengers historically had a lower survival rate.")

    if pclass == 1:
        analysis.append("🎟 First-class passengers generally had better access to lifeboats.")

    elif pclass == 3:
        analysis.append("🎟 Third-class passengers generally had lower survival rates.")

    if age < 12:
        analysis.append("🧒 Children were often prioritized during evacuation.")

    elif age > 60:
        analysis.append("👴 Elderly passengers generally faced greater evacuation challenges.")

    if fare > 100:
        analysis.append("💰 A higher fare often corresponded to higher-class accommodation.")

    if sibsp > 2:
        analysis.append("👨‍👩‍👧 A larger family group could affect evacuation dynamics.")

    for item in analysis:
        st.write(item)
# ==========================================================
# ANALYTICS DASHBOARD
# ==========================================================

st.markdown("---")

st.header("📈 Dashboard Analytics")

a1, a2, a3 = st.columns(3)

with a1:
    st.metric(
        "🚢 Passenger Class",
        pclass
    )

with a2:
    st.metric(
        "🎂 Passenger Age",
        age
    )

with a3:
    st.metric(
        "💰 Ticket Fare",
        f"${fare:.2f}"
    )

st.markdown("---")

# ==========================================================
# PASSENGER VISUAL SUMMARY
# ==========================================================

st.subheader("📊 Passenger Profile")

profile = pd.DataFrame({
    "Feature":[
        "Passenger Class",
        "Age",
        "Fare",
        "Siblings",
        "Parents"
    ],
    "Value":[
        pclass,
        age,
        fare,
        sibsp,
        parch
    ]
})

st.bar_chart(profile.set_index("Feature"))

# ==========================================================
# DATASET INFORMATION
# ==========================================================

with st.expander("📚 About Titanic Dataset", expanded=False):

    st.markdown("""
### 🚢 Titanic Survival Dataset

The Titanic dataset is one of the most famous machine learning datasets.

### Features

- Passenger Class
- Gender
- Age
- Fare
- Siblings / Spouse
- Parents / Children
- Embarked Port

### Target

- **0 → Did Not Survive**
- **1 → Survived**

### Machine Learning Model

- Random Forest Classifier

### Accuracy

Approximately **80–85%**

### Number of Passengers

**891**

### Prediction Goal

Predict whether a passenger would survive based on historical passenger information.
""")

# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.markdown("---")

st.header("🤖 Model Information")

model_df = pd.DataFrame({

"Property":[

"Algorithm",

"Accuracy",

"Features",

"Dataset Size"

],

"Value":[

info["model_type"],

f"{info['accuracy']*100:.2f}%",

len(info["feature_names"]),

891

]

})

st.dataframe(

model_df,

hide_index=True,

use_container_width=True

)

# ==========================================================
# FEATURE DESCRIPTION
# ==========================================================

with st.expander("📖 Feature Description"):

    st.markdown("""

| Feature | Description |
|---------|-------------|
| Pclass | Passenger Class |
| Sex | Gender |
| Age | Passenger Age |
| SibSp | Number of Siblings/Spouse |
| Parch | Number of Parents/Children |
| Fare | Ticket Fare |
| Embarked | Boarding Port |

""")

# ==========================================================
# RANDOM FACTS
# ==========================================================

st.markdown("---")

st.header("🚢 Titanic Facts")

facts = [

"🚢 RMS Titanic sank on 15 April 1912.",

"👥 Around 2,224 passengers and crew were onboard.",

"💔 More than 1,500 people lost their lives.",

"👩 Women and children generally had higher survival rates.",

"🎟 First-class passengers survived more often than third-class passengers."

]

for fact in facts:

    st.info(fact)

# ==========================================================
# TECHNOLOGY STACK
# ==========================================================

st.markdown("---")

st.header("🛠 Technology Stack")

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.success("🐍 Python")

with c2:
    st.success("📊 Streamlit")

with c3:
    st.success("🤖 Scikit-Learn")

with c4:
    st.success("🐼 Pandas")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown("""

<div style='text-align:center;
padding:25px;
border-radius:15px;
background:linear-gradient(90deg,#005AA7,#00CDAC);
color:white;
font-size:18px;'>

<h2>🚢 Titanic Survival Prediction Dashboard</h2>

<p>
Built using
<strong>Python • Streamlit • Scikit-Learn • Pandas • NumPy</strong>
</p>

<p>
Developed for Machine Learning Deployment
</p>

<p>
❤️ Designed with Streamlit
</p>

</div>

""", unsafe_allow_html=True)    