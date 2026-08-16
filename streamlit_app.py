import joblib
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Mental Health Score Predictor", page_icon="🧠", layout="centered")

MODEL_PATH = "Mental_Health_Model.pkl"

top_countries = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK',
                  'Germany', 'Mexico', 'Turkey', 'France']

genders            = ['Male', 'Female']
academic_levels    = ['Undergraduate', 'Graduate', 'High School']
platforms          = ['Facebook', 'LinkedIn', 'Instagram', 'Snapchat', 'Twitter',
                       'YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte',
                       'WhatsApp', 'WeChat']
purposes           = ['Networking', 'Education', 'Entertainment', 'News']
stress_levels      = ['Low', 'Medium', 'High', 'Very High']


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("🧠 Mental Health Score Predictor")
st.write(
    "Fill in the details below to estimate a student's mental health score "
    "based on lifestyle and social media usage patterns."
)

with st.form("prediction_form"):
    st.subheader("Personal Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=20, step=1)
        gender = st.selectbox("Gender", genders)
    with col2:
        country = st.text_input("Country", value="India")
        academic_level = st.selectbox("Academic Level", academic_levels)

    st.subheader("Social Media Usage")
    col3, col4 = st.columns(2)
    with col3:
        most_used_platform = st.selectbox("Most Used Platform", platforms)
        purpose_of_use = st.selectbox("Purpose of Use", purposes)
    with col4:
        avg_daily_usage_hours = st.slider("Avg Daily Usage (hours)", 0.0, 24.0, 3.0, 0.1)
        daily_unlocks = st.number_input("Daily Phone Unlocks", min_value=0, value=50, step=1)

    st.subheader("Lifestyle")
    col5, col6, col7 = st.columns(3)
    with col5:
        study_hours = st.slider("Study Hours/day", 0.0, 24.0, 4.0, 0.1)
    with col6:
        physical_activity_hours = st.slider("Physical Activity Hours/day", 0.0, 24.0, 1.0, 0.1)
    with col7:
        sleep_hours_per_night = st.slider("Sleep Hours/night", 0.0, 24.0, 7.0, 0.1)

    stress_level = st.select_slider("Stress Level", options=stress_levels, value="Medium")

    submitted = st.form_submit_button("Predict Mental Health Score")

if submitted:
    country_group = country if country in top_countries else "Other"

    input_row = pd.DataFrame([{
        'Age':                     age,
        'Gender':                  gender,
        'Country':                 country,
        'Academic_Level':          academic_level,
        'Most_Used_Platform':      most_used_platform,
        'Purpose_Of_Use':          purpose_of_use,
        'Avg_Daily_Usage_Hours':   avg_daily_usage_hours,
        'Daily_Unlocks':           daily_unlocks,
        'Study_Hours':             study_hours,
        'Physical_Activity_Hours': physical_activity_hours,
        'Sleep_Hours_Per_Night':   sleep_hours_per_night,
        'Stress_Level':            stress_level,
        'Grouped_country':         country_group,
    }])

    try:
        prediction = model.predict(input_row)[0]
        score = round(float(prediction), 2)

        st.success(f"### Predicted Mental Health Score: **{score}**")

        if score >= 7:
            st.info("This falls in a generally healthy range. Keep up the good habits! 🌿")
        elif score >= 4:
            st.warning("This is a moderate range — small lifestyle tweaks (sleep, activity) could help. ⚖️")
        else:
            st.error(
                "This score suggests notable strain. Please consider talking to a "
                "counselor, trusted friend, or mental health professional. 💙"
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption(
    "Note: This tool provides an estimate from a machine learning model trained on "
    "survey data. It is not a diagnostic tool and does not replace professional advice."
)
