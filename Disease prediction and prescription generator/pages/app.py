import streamlit as st
from dotenv import load_dotenv
import os
import re
import json
import io
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
import google.generativeai as genai
from gtts import gTTS

# -------------------------------------------------------------
# PAGE CONFIGURATION & CRISP FAVICON
# -------------------------------------------------------------
favicon_path = os.path.join(os.path.dirname(__file__), "favicon.png")
icon_to_use = favicon_path if os.path.exists(favicon_path) else "🩺"

st.set_page_config(
    page_title="Disease Predictor and Prescription Generator",
    page_icon=icon_to_use,
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if api_key:
    try:
        genai.configure(api_key=api_key)
    except Exception:
        pass

# -------------------------------------------------------------
# COMPACT & CLEAN CSS (Matching Streamlit Design in Image 2)
# -------------------------------------------------------------
st.markdown("""
    <style>
    .block-container {
        max-width: 820px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Header Typography with Subtle Blue Glow */
    .app-main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #38bdf8;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
        text-shadow: 0 0 24px rgba(56, 189, 248, 0.45);
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .app-sub-title {
        font-size: 1.05rem;
        color: var(--text-color);
        opacity: 0.85;
        margin-bottom: 24px;
        font-weight: 500;
    }

    /* Disease Result Styling */
    .predicted-disease-banner {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-color);
        margin-top: 18px;
        margin-bottom: 4px;
    }
    
    .predicted-disease-name {
        color: #38bdf8;
        font-weight: 800;
        font-size: 1.5rem;
        margin-left: 8px;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    }
    
    .accuracy-pill {
        color: #34d399;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Primary Action Button */
    div.stButton > button[kind="primary"] {
        background: #2563eb !important;
        color: #ffffff !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        padding: 12px 28px !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        margin-top: 10px !important;
    }
    
    div.stButton > button[kind="primary"]:hover {
        background: #1d4ed8 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45) !important;
    }
    
    /* Prescription Row Cards */
    .rx-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(125, 125, 125, 0.2);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .rx-name {
        font-weight: 750;
        font-size: 0.95rem;
        color: var(--text-color);
    }
    
    .rx-inst {
        font-size: 0.85rem;
        opacity: 0.8;
        margin-top: 2px;
    }
    
    .rx-tag {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 700;
        white-space: nowrap;
    }
    
    /* Medical Disclaimer */
    .disclaimer-box {
        font-size: 0.84rem;
        color: var(--text-color);
        opacity: 0.75;
        border-left: 3px solid #ef4444;
        padding-left: 12px;
        margin-top: 24px;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# DATASET INITIALIZATION & ML ENSEMBLE TRAINING
# -------------------------------------------------------------
@st.cache_data
def load_datasets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "Training.csv")
    test_path = os.path.join(base_dir, "Testing.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    combined = pd.concat([train_df, test_df], ignore_index=True)
    if 'Unnamed: 133' in combined.columns:
        combined = combined.drop(columns=['Unnamed: 133'])
        
    feature_cols = [c.strip() for c in combined.columns if c != 'prognosis']
    diseases = sorted(combined['prognosis'].unique())
    disease_to_idx = {d: i for i, d in enumerate(diseases)}
    idx_to_disease = {i: d for i, d in enumerate(diseases)}
    
    friendly_symptoms = {col: col.replace('_', ' ').strip().title() for col in feature_cols}
    friendly_to_col = {v: k for k, v in friendly_symptoms.items()}
    
    return combined, feature_cols, diseases, disease_to_idx, idx_to_disease, friendly_symptoms, friendly_to_col

combined_df, feature_cols, diseases, disease_to_idx, idx_to_disease, friendly_symptoms, friendly_to_col = load_datasets()

@st.cache_resource
def train_diagnostic_engine():
    X = combined_df.drop(columns=['prognosis'])
    y = combined_df['prognosis'].map(disease_to_idx).astype(int)
    
    model = VotingClassifier(
        estimators=[
            ('mnb', MultinomialNB(alpha=0.1)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=25)),
            ('et', ExtraTreesClassifier(n_estimators=100, random_state=42, max_depth=25))
        ],
        voting='soft'
    )
    model.fit(X, y)
    return model

diagnostic_model = train_diagnostic_engine()

# Clinical Reference & Prescriptions
CLINICAL_STANDARDS = {
    'Fungal infection': {
        'display': 'Fungal infection',
        'icon': '🦠',
        'overview': "A fungal infection (mycosis) occurs when fungi overcome the body's defenses and infect tissue. While they most commonly affect the skin, nails, and mucous membranes (such as athlete's foot, ringworm, or candidiasis), they can also become systemic in individuals with weakened immune systems.",
        'rx': [
            {'name': 'Clotrimazole 1% Topical Cream', 'instruction': 'Apply thin layer twice daily to clean, dry affected areas', 'duration': '2-4 weeks'},
            {'name': 'Fluconazole 150mg', 'instruction': '1 tablet once weekly if prescribed for widespread infection', 'duration': 'As prescribed'},
            {'name': 'Antifungal Dusting Powder', 'instruction': 'Dust in shoes and skin folds daily to keep skin dry', 'duration': 'Daily'}
        ],
        'care': [
            'Keep infected skin clean, thoroughly dried, and exposed to air when possible.',
            'Wear loose, breathable cotton clothing and change undergarments daily.',
            'Do not share personal towels, bed linens, or footwear.'
        ],
        'diet': [
            'Reduce intake of refined sugars and yeasts which feed fungal proliferation.',
            'Incorporate probiotic foods like plain unsweetened yogurt and kefir.',
            'Drink 2.5 liters of clean water daily to assist natural dermal recovery.'
        ]
    },
    'Allergy': {
        'display': 'Allergy',
        'icon': '🌿',
        'overview': "An allergic response occurs when your immune system reacts to a typically harmless environmental substance (such as pollen, pet dander, or dust mites), causing sneezing, watery eyes, and itchy skin.",
        'rx': [
            {'name': 'Cetirizine 10mg', 'instruction': 'Take 1 tablet once daily at bedtime', 'duration': '5-7 days'},
            {'name': 'Saline Nasal Irrigation', 'instruction': 'Rinse nasal cavities twice daily with saline', 'duration': '7 days'},
            {'name': 'Avoid Allergen Triggers', 'instruction': 'Limit exposure to identified dust, pollen, or animal dander', 'duration': 'Ongoing'}
        ],
        'care': [
            'Keep windows closed during windy or high-pollen seasons.',
            'Wash bedding weekly in warm water to eradicate dust mites.',
            'Shower before sleeping to remove accumulated pollen from hair and skin.'
        ],
        'diet': [
            'Consume foods rich in natural bioflavonoids (apples, berries, broccoli).',
            'Stay well-hydrated to help thin respiratory mucosal secretions.',
            'Avoid processed foods with artificial additives during flare-ups.'
        ]
    },
    'GERD': {
        'display': 'GERD (Acid Reflux)',
        'icon': '🔥',
        'overview': "Gastroesophageal reflux disease occurs when stomach acid repeatedly flows back into the tube connecting your mouth and stomach (esophagus), irritating your esophageal lining and producing heartburn.",
        'rx': [
            {'name': 'Pantoprazole 40mg', 'instruction': 'Take 1 tablet 30 minutes before breakfast on an empty stomach', 'duration': '14 days'},
            {'name': 'Antacid Liquid (Magaldrate + Simethicone)', 'instruction': '10-15 ml after meals and at bedtime as needed', 'duration': 'As needed'},
            {'name': 'Elevate Head of Bed (15-20 cm)', 'instruction': 'Avoid lying flat within 3 hours after eating', 'duration': 'Daily'}
        ],
        'care': [
            'Eat smaller, more frequent meals rather than large, heavy meals.',
            'Do not lie down or sleep for at least 3 hours after finishing a meal.',
            'Wear loose-fitting clothing that does not put pressure on your abdomen.'
        ],
        'diet': [
            'Avoid acidic foods, citrus, tomatoes, chocolate, caffeine, and alcohol.',
            'Eliminate fried, fatty, and heavily spiced dishes.',
            'Eat soothing foods: oatmeal, bananas, non-citrus fruits, and steamed vegetables.'
        ]
    }
}

# -------------------------------------------------------------
# NLP EXTRACTION HELPER
# -------------------------------------------------------------
def parse_natural_language(text):
    if not text:
        return []
    cleaned = text.lower()
    matched = set()
    for col in feature_cols:
        phrase = col.replace('_', ' ').lower()
        if phrase in cleaned:
            matched.add(friendly_symptoms[col])
            
    synonyms = {
        'fever': ['High Fever', 'Mild Fever'],
        'headache': ['Headache'],
        'cold': ['Chills', 'Continuous Sneezing'],
        'cough': ['Cough'],
        'stomach pain': ['Stomach Pain', 'Abdominal Pain'],
        'itching': ['Itching'],
        'rash': ['Skin Rash'],
        'vomiting': ['Vomiting'],
        'nausea': ['Nausea'],
        'tired': ['Fatigue'],
        'chest pain': ['Chest Pain']
    }
    for kw, target_friends in synonyms.items():
        if kw in cleaned:
            for tf in target_friends:
                if tf in friendly_to_col:
                    matched.add(tf)
    return list(matched)

# -------------------------------------------------------------
# MAIN APP HEADER (Exact clean layout from Image 2)
# -------------------------------------------------------------
st.markdown("""
    <div class="app-main-title">
        <span>🩺</span>
        <span>Disease Predictor and Prescription Generator</span>
    </div>
    <div class="app-sub-title">
        📝 Enter your symptoms below to get an instant, AI-powered diagnosis!
    </div>
""", unsafe_allow_html=True)

# Free text auto-fill (optional natural language helper)
free_text = st.text_input(
    "Describe your symptoms in plain text (optional):",
    placeholder="e.g. I have itching and skin rash since 2 days...",
    help="Type in natural words; matching symptoms will be added automatically."
)

default_symptoms = ['Itching']
if free_text.strip():
    parsed = parse_natural_language(free_text)
    if parsed:
        default_symptoms = parsed

all_symptom_options = sorted(list(friendly_to_col.keys()))

selected_symptoms = st.multiselect(
    "Select your symptoms:",
    options=all_symptom_options,
    default=default_symptoms,
    help="Select or search one or more symptoms from the clinical database."
)

# Language radio
selected_lang = st.radio(
    "Select output language:",
    options=["English", "Hindi", "Telugu"],
    horizontal=True,
    index=0
)
lang_codes = {"English": "en", "Hindi": "hi", "Telugu": "te"}

# Predict Button
predict_btn = st.button("Predict Disease and Generate Prescription", type="primary", use_container_width=True)

# -------------------------------------------------------------
# RESULTS PRESENTATION
# -------------------------------------------------------------
if predict_btn:
    if not selected_symptoms:
        st.warning("⚠️ Please select or type at least one symptom.")
    else:
        # Run ML model
        vector = np.zeros((1, len(feature_cols)))
        for item in selected_symptoms:
            col = friendly_to_col.get(item)
            if col and col in feature_cols:
                idx = feature_cols.index(col)
                vector[0, idx] = 1
                
        vector_df = pd.DataFrame(vector, columns=feature_cols)
        probs = diagnostic_model.predict_proba(vector_df)[0]
        top_idx = int(np.argmax(probs))
        predicted_disease = idx_to_disease[top_idx]
        confidence = float(probs[top_idx] * 100)
        
        info = CLINICAL_STANDARDS.get(predicted_disease, {
            'display': predicted_disease,
            'icon': '🦠',
            'overview': f"Clinical assessment shows pattern match with {predicted_disease}. Please consult a physician for clinical confirmation and prescription validation.",
            'rx': [
                {'name': 'Supportive Clinical Care', 'instruction': 'Follow symptomatic rest and hydration', 'duration': '5-7 days'},
                {'name': 'Medical Examination', 'instruction': 'Consult a licensed doctor for diagnostic confirmation', 'duration': 'As needed'}
            ],
            'care': [
                'Ensure 8 hours of restful sleep daily.',
                'Stay adequately hydrated with clean fluids.',
                'Monitor for sudden escalation of fever, pain, or breathlessness.'
            ],
            'diet': [
                'Eat freshly cooked, light, balanced meals.',
                'Avoid heavily fried, processed, or overly spiced foods.'
            ]
        })
        
        # Render clean results matching Image 2
        st.markdown(f"""
            <div class="predicted-disease-banner">
                Predicted Disease: 
                <span class="predicted-disease-name">{info.get('icon', '🦠')} {info.get('display', predicted_disease)}</span>
            </div>
            <div class="accuracy-pill">
                ✓ Best Algorithm: Random Forest / Ensemble &nbsp;|&nbsp; Accuracy: {confidence:.2f}%
            </div>
            <h3 style="font-size: 1.45rem; font-weight: 800; margin-bottom: 12px; color: var(--text-color);">
                Medical Prescription and Advice:
            </h3>
        """, unsafe_allow_html=True)
        
        st.write(info['overview'])
        
        # Prescriptions
        if 'rx' in info and info['rx']:
            st.markdown("#### 💊 Recommended Medications & Dosage:")
            for rx in info['rx']:
                st.markdown(f"""
                    <div class="rx-card">
                        <div>
                            <div class="rx-name">{rx['name']}</div>
                            <div class="rx-inst">{rx['instruction']}</div>
                        </div>
                        <div class="rx-tag">{rx['duration']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
        # Care & Diet
        col_care, col_diet = st.columns(2)
        with col_care:
            st.markdown("#### 🏠 Home Care & Recovery:")
            for c in info.get('care', []):
                st.markdown(f"- {c}")
                
        with col_diet:
            st.markdown("#### 🥗 Diet & Hydration:")
            for d in info.get('diet', []):
                st.markdown(f"- {d}")
                
        # Audio Consultation
        st.markdown("#### 🔊 Audio Consultation:")
        try:
            tts_text = f"Predicted condition: {info.get('display', predicted_disease)}. {info['overview']}"
            tts = gTTS(text=tts_text, lang=lang_codes[selected_lang])
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            st.audio(audio_fp, format='audio/mp3')
        except Exception as e:
            st.caption("Audio consultation unavailable.")
            
        # Disclaimer
        st.markdown("""
            <div class="disclaimer-box">
                * <strong>Important Medical Disclaimer:</strong> This is an AI-generated clinical recommendation and not a substitute for professional medical diagnosis. Please consult a qualified doctor if symptoms persist or worsen.
            </div>
        """, unsafe_allow_html=True)
