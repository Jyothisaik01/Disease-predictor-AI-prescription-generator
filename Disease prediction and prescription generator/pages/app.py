import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
from collections import Counter
import google.generativeai as genai
from googletrans import Translator
from gtts import gTTS
import io

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load datasets
train_data = pd.read_csv("Training.csv")
test_data = pd.read_csv("Testing.csv")

# Define symptoms and diseases
symptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 
    'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 
    'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 
    'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 
    'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 
    'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 
    'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes',
    'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 
    'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 
    'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 
    'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 
    'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain',
    'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 
    'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',  
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 
    'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 
    'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 
    'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 
    'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 
    'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 
    'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]

diseases = train_data['prognosis'].unique()

# Replace disease names with numeric codes for model training
train_data.replace({'prognosis': {disease: idx for idx, disease in enumerate(diseases)}}, inplace=True)
test_data.replace({'prognosis': {disease: idx for idx, disease in enumerate(diseases)}}, inplace=True)

# Prepare features and target
X = train_data.drop(['prognosis', 'Unnamed: 133'], axis=1, errors='ignore')
y = train_data['prognosis']

# Save features used for training
with open('symptom_features.txt', 'w') as f:
    for col in X.columns:
        f.write(f"{col}\n")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Selection and Training ---
def train_best_model(X_train, y_train, X_test, y_test):
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB()
    }
    best_model = None
    best_acc = 0
    best_name = ''
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
    joblib.dump(best_model, 'disease_predictor_model.joblib')
    return best_model, best_name, best_acc

# Train and select best model
best_model, best_model_name, best_model_acc = train_best_model(X_train, y_train, X_test, y_test)

# Prediction function
def predict_disease(input_symptoms):
    # Load features in the correct order
    with open('symptom_features.txt', 'r') as f:
        symptoms = [line.strip() for line in f.readlines()]
    input_vector = [1 if symptom in input_symptoms else 0 for symptom in symptoms]
    input_vector = np.array(input_vector).reshape(1, -1)

    # Load trained model
    model = joblib.load('disease_predictor_model.joblib')

    # Get prediction
    prediction = model.predict(input_vector)[0]

    # Get disease name from prediction
    predicted_disease = diseases[prediction]
    return predicted_disease

# AI-generated medical prescription using Gemini
def generate_medical_prescription(disease):
    """Generate medical prescription, causes, and dietary recommendations for the predicted disease."""
    prompt = f"""
    You are a medical assistant. Provide detailed information for the disease '{disease}', including:
    - Causes of the disease.
    - Recommended medical prescriptions.
    - Food and dietary advice.
    Ensure the response is clear and concise.
    """
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return response.text

# Streamlit App UI
st.set_page_config(page_title="Disease Predictor", page_icon="🩺", layout="centered", initial_sidebar_state="collapsed")

# --- UI Animations and Styling ---
st.markdown("""
    <style>
    .animated-title {
        font-size: 2.2rem;
        font-weight: bold;
        # color: #fff;
        # text-shadow: 2px 2px 10px rgba(179,224,255,0.35), 0 2px 8px rgba(0,91,181,0.18);
        color: #0072ff;
        text-shadow: 2px 2px 10px rgba(179,224,255,0.35), 0 2px 8px rgba(0,91,181,0.18);
        animation: pop 1.1s ease;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
    }
    .animated-subheader {
        font-size: 1.2rem;
        color: #fff;
        text-shadow: 1px 1px 8px rgba(204,230,255,0.28);
        animation: fadein 2s;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .stButton > button {
        background-color: #0072ff;
        color: #fff;
        border: none;
        border-radius: 30px;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.7em 2em;
        margin: 10px 0;
        box-shadow: 0 2px 12px rgba(0,114,255,0.09);
        transition: background 0.2s, transform 0.2s, box-shadow 0.2s;
        outline: none;
        cursor: pointer;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #0072ff 60%, #00c6ff 100%);
        color: #fff;
        transform: translateY(-2px) scale(1.03);
        box-shadow: 0 4px 24px rgba(0,114,255,0.18);
    }
    .stDownloadButton > button, .download-btn {
        background: linear-gradient(90deg, #00c66c 60%, #00ffb8 100%);
        color: #fff;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 12px 32px;
        border-radius: 30px;
        border: none;
        box-shadow: 0 2px 12px rgba(0,198,108,0.10);
        cursor: pointer;
        transition: background 0.2s, transform 0.2s, box-shadow 0.2s;
        outline: none;
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 8px 0;
    }
    .stDownloadButton > button:hover, .download-btn:hover {
        background: linear-gradient(90deg, #00ffb8 60%, #00c66c 100%);
        color: #fff;
        transform: scale(1.04) translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,198,108,0.16);
    }
    @keyframes pop {
        0% { transform: scale(0.7); opacity: 0; }
        70% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(1); }
    }
    @keyframes fadein {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    /* Animation for the Select your symptoms bar */
    .symptom-bar-animate .stMultiSelect, .symptom-bar-animate .stMultiSelect label {
        animation: symptomFadePop 1.2s cubic-bezier(0.23, 1, 0.32, 1);
    }
    @keyframes symptomFadePop {
        0% { transform: scale(0.7); opacity: 0; }
        70% { transform: scale(1.05); opacity: 1; }
        100% { transform: scale(1); }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="animated-title">🩺 Disease Predictor and Prescription Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="animated-subheader">📝 Enter your symptoms below to get an instant, AI-powered diagnosis!</div>', unsafe_allow_html=True)

# Input fields for symptoms
st.markdown('<div class="symptom-bar-animate">', unsafe_allow_html=True)
selected_symptoms = st.multiselect("Select your symptoms:", options=symptoms, help="Search and select your symptoms.")
st.markdown('</div>', unsafe_allow_html=True)

# Language selection
lang_map = {"English": "en", "Hindi": "hi", "Telugu": "te"}
language = st.radio("Select output language:", options=list(lang_map.keys()), horizontal=True)
lang_code = lang_map[language]

# Prediction and Medical Prescription
if st.button("Predict Disease and Generate Prescription"):
    if selected_symptoms:
        with st.spinner("Predicting disease..."):
            result = predict_disease(selected_symptoms)
            st.session_state['result'] = result
            st.markdown(f"Predicted Disease: <span style='color:#0072ff; font-weight:bold; text-shadow:1px 1px 8px rgba(179,224,255,0.28); font-size:1.2em;'>🦠 {result}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-weight:bold; color:#009e60; text-shadow:0 1px 6px rgba(179,224,255,0.18);'>✔️ Best Algorithm: {best_model_name} &nbsp; | &nbsp; Accuracy: {best_model_acc * 100:.2f}%</span>", unsafe_allow_html=True)

        with st.spinner("Generating medical prescription..."):
            prescription = generate_medical_prescription(result)

        if prescription:
            # Remove asterisks
            prescription_clean = prescription.replace("*", "")
            # Translate if not English
            if lang_code != "en":
                translator = Translator()
                translated = translator.translate(prescription_clean, dest=lang_code)
                prescription_clean = translated.text
            # Save to session state
            st.session_state['prescription_clean'] = prescription_clean
            # Generate audio file
            tts = gTTS(text=prescription_clean, lang=lang_code)
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            st.session_state['audio_fp'] = audio_fp
            st.session_state['lang_code'] = lang_code
    else:
        st.error("Please select at least one symptom.")

# Display results and download buttons if available
if 'prescription_clean' in st.session_state and 'audio_fp' in st.session_state:
    st.markdown("## Medical Prescription and Advice:")
    st.write(st.session_state['prescription_clean'])
    st.audio(st.session_state['audio_fp'], format="audio/mp3")
    st.download_button(
        label="Download Prescription Audio",
        data=st.session_state['audio_fp'],
        file_name=f"{st.session_state.get('result', 'prescription')}_{st.session_state.get('lang_code', 'en')}.mp3",
        mime="audio/mp3"
    )
    st.download_button(
        label="Download Prescription Text",
        data=st.session_state['prescription_clean'],
        file_name=f"{st.session_state.get('result', 'prescription')}_{st.session_state.get('lang_code', 'en')}.txt",
        mime="text/plain"
    )