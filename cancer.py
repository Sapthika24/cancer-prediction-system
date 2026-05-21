# ==========================================
# MASTER AI MULTI-CANCER PREDICTION SYSTEM
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Cancer Screening System", layout="wide")
st.title("🏥 Advanced AI Multi-Cancer Screening System")

# ---------------------------
# Load Dataset
# ---------------------------
data = pd.read_csv("multi_cancer_dataset.csv")

# Clean possible whitespace issues
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# ---------------------------
# Encode Categorical Columns
# ---------------------------
categorical_cols = [
    "gender","location_type","occupation_risk",
    "alcohol_frequency","physical_activity_level",
    "diet_quality","pollution_exposure","severity"
]

label_encoders = {}

for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
data["cancer_type"] = target_encoder.fit_transform(data["cancer_type"])

# ---------------------------
# Split Data
# ---------------------------
X = data.drop("cancer_type", axis=1)
y = data["cancer_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

st.sidebar.header("Model Performance")
st.sidebar.write(f"Accuracy: {accuracy*100:.2f}%")

# ---------------------------
# USER INPUT SECTION
# ---------------------------
st.header("Enter Patient Information")

col1, col2 = st.columns(2)

# Use dataset-trained options (IMPORTANT FIX)
gender_options = label_encoders["gender"].classes_
location_options = label_encoders["location_type"].classes_
occupation_options = label_encoders["occupation_risk"].classes_
alcohol_options = label_encoders["alcohol_frequency"].classes_
activity_options = label_encoders["physical_activity_level"].classes_
diet_options = label_encoders["diet_quality"].classes_
pollution_options = label_encoders["pollution_exposure"].classes_
severity_options = label_encoders["severity"].classes_

with col1:
    age = st.slider("Age", 18, 85, 30)
    gender = st.selectbox("Gender", gender_options)
    location = st.selectbox("Location Type", location_options)
    occupation = st.selectbox("Occupation Risk", occupation_options)
    family = st.checkbox("Family History of Cancer")
    genetic = st.checkbox("Genetic Mutation")
    previous = st.checkbox("Previous Cancer History")
    smoking_years = st.slider("Smoking Years", 0, 30, 0)
    alcohol = st.selectbox("Alcohol Frequency", alcohol_options)
    bmi = st.slider("BMI", 18, 40, 22)

with col2:
    activity = st.selectbox("Physical Activity", activity_options)
    diet = st.selectbox("Diet Quality", diet_options)
    pollution = st.selectbox("Pollution Exposure", pollution_options)
    radiation = st.checkbox("Radiation Exposure")
    chemical = st.checkbox("Chemical Exposure")
    lump = st.checkbox("Lump in Breast")
    chest = st.checkbox("Chest Pain")
    mole = st.checkbox("Mole Change")
    weight = st.checkbox("Weight Loss")
    cough = st.checkbox("Persistent Cough")
    fatigue = st.checkbox("Fatigue")
    night = st.checkbox("Night Sweats")
    swallow = st.checkbox("Difficulty Swallowing")
    duration = st.slider("Symptom Duration (Months)", 1, 24, 3)
    severity = st.selectbox("Severity Level", severity_options)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Cancer Risk"):

    input_dict = {
        "age": age,
        "gender": label_encoders["gender"].transform([gender])[0],
        "location_type": label_encoders["location_type"].transform([location])[0],
        "occupation_risk": label_encoders["occupation_risk"].transform([occupation])[0],
        "family_history": int(family),
        "genetic_mutation": int(genetic),
        "previous_cancer_history": int(previous),
        "smoking_years": smoking_years,
        "alcohol_frequency": label_encoders["alcohol_frequency"].transform([alcohol])[0],
        "physical_activity_level": label_encoders["physical_activity_level"].transform([activity])[0],
        "diet_quality": label_encoders["diet_quality"].transform([diet])[0],
        "bmi": bmi,
        "pollution_exposure": label_encoders["pollution_exposure"].transform([pollution])[0],
        "radiation_exposure": int(radiation),
        "chemical_exposure": int(chemical),
        "lump_breast": int(lump),
        "chest_pain": int(chest),
        "mole_change": int(mole),
        "weight_loss": int(weight),
        "persistent_cough": int(cough),
        "fatigue": int(fatigue),
        "night_sweats": int(night),
        "difficulty_swallowing": int(swallow),
        "symptom_duration_months": duration,
        "severity": label_encoders["severity"].transform([severity])[0]
    }

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    result = target_encoder.inverse_transform([prediction])[0]

    st.subheader("Prediction Result")

    if result == "Low Risk":
        st.success("Low Cancer Risk Detected")
    else:
        st.error(f"High Risk: {result}")

    # Probability chart
    prob_df = pd.DataFrame({
        "Cancer Type": target_encoder.classes_,
        "Probability": probabilities
    })

    st.subheader("Prediction Probabilities")
    st.bar_chart(prob_df.set_index("Cancer Type"))

    # Feature importance
    st.subheader("Top Influencing Factors")

    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(10)

    st.bar_chart(feature_df.set_index("Feature"))