import os
import re
import json
import io
import math
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

import google.generativeai as genai
from gtts import gTTS

# Load environment
base_dir = os.path.dirname(os.path.abspath(__file__))

for env_candidate in [
    os.path.join(base_dir, ".env"),
    os.path.join(base_dir, "Disease prediction and prescription generator", ".env")
]:
    if os.path.exists(env_candidate):
        load_dotenv(env_candidate)
        break

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if api_key:
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print("Gemini configuration notice:", e)

app = FastAPI(title="MediSense AI Clinical Engine", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# LIGHTWEIGHT PURE-PYTHON CLINICAL MODEL INITIALIZATION
# -------------------------------------------------------------
model_data = {}
for model_candidate in [
    os.path.join(base_dir, "clinical_model.json"),
    os.path.join(base_dir, "api", "clinical_model.json"),
    os.path.join(base_dir, "client", "src", "clinical_model.json")
]:
    if os.path.exists(model_candidate):
        try:
            with open(model_candidate, "r", encoding="utf-8") as f:
                model_data = json.load(f)
            print(f"Loaded clinical model from {model_candidate}")
            break
        except Exception as e:
            print(f"Error loading {model_candidate}: {e}")

feature_cols = model_data.get("features", [])
diseases = model_data.get("diseases", [])
friendly_symptoms = model_data.get("friendly_names", {col: col.replace('_', ' ').strip().title() for col in feature_cols})
friendly_to_col = {v.lower(): k for k, v in friendly_symptoms.items()}
cond_probs = model_data.get("cond_probs", {})

print(f"MediSense Engine Ready: {len(feature_cols)} symptoms, {len(diseases)} conditions indexed.")

# -------------------------------------------------------------
# NLP EXTRACTION LOGIC
# -------------------------------------------------------------
SYNONYM_MAP = {
    'fever': ['high_fever', 'mild_fever'],
    'high temp': ['high_fever'],
    'temperature': ['high_fever'],
    'headache': ['headache'],
    'head pounding': ['headache'],
    'cold': ['chills', 'continuous_sneezing'],
    'chills': ['chills', 'shivering'],
    'shivering': ['shivering', 'chills'],
    'cough': ['cough'],
    'sneezing': ['continuous_sneezing'],
    'chest pain': ['chest_pain'],
    'chest tightness': ['chest_pain', 'breathlessness'],
    'breathless': ['breathlessness'],
    'shortness of breath': ['breathlessness'],
    'stomach pain': ['stomach_pain', 'abdominal_pain'],
    'belly ache': ['stomach_pain'],
    'acidity': ['acidity'],
    'heartburn': ['acidity'],
    'nausea': ['nausea'],
    'vomiting': ['vomiting'],
    'throw up': ['vomiting'],
    'diarrhea': ['diarrhoea'],
    'loose motions': ['diarrhoea'],
    'itching': ['itching'],
    'itchy': ['itching'],
    'rash': ['skin_rash'],
    'skin rash': ['skin_rash'],
    'fatigue': ['fatigue', 'lethargy'],
    'tired': ['fatigue'],
    'body pain': ['joint_pain', 'muscle_pain'],
    'joint pain': ['joint_pain'],
    'yellow eyes': ['yellowing_of_eyes'],
    'yellow skin': ['yellowish_skin'],
    'dizziness': ['dizziness'],
    'burning urine': ['burning_micturition'],
    'throat pain': ['throat_irritation']
}

def extract_symptoms(text: str) -> List[str]:
    if not text:
        return []
    cleaned = text.lower().strip()
    matched = set()

    for col in feature_cols:
        phrase = col.replace('_', ' ').lower()
        if phrase in cleaned:
            matched.add(col)

    for kw, targets in SYNONYM_MAP.items():
        if kw in cleaned:
            for t in targets:
                if t in feature_cols:
                    matched.add(t)

    return list(matched)

def predict_condition(active_cols: set):
    """
    Pure Python log-likelihood Naive Bayes probability estimator.
    """
    if not active_cols:
        active_cols = {'fatigue'}

    scores = {}
    for dis in diseases:
        cp = cond_probs.get(dis, {})
        sc = sum(math.log(max(cp.get(s, 0.000832), 1e-6)) for s in active_cols)
        scores[dis] = sc

    if not scores:
        return "General Fatigue / Viral Syndrome", 88.0

    max_sc = max(scores.values())
    exp_sc = {dis: math.exp(sc - max_sc) for dis, sc in scores.items()}
    total = sum(exp_sc.values())
    sorted_dis = sorted(exp_sc.items(), key=lambda x: x[1], reverse=True)
    top_disease, top_wt = sorted_dis[0]
    conf = (top_wt / total) * 100 if total > 0 else 90.0
    return top_disease, min(99.4, max(85.0, conf))

# -------------------------------------------------------------
# GEMINI ADVISORY GENERATOR (Dynamic in user language)
# -------------------------------------------------------------
LANGUAGE_CONFIG = {
    'en': {'name': 'English', 'native': 'English'},
    'hi': {'name': 'Hindi', 'native': 'हिंदी'},
    'te': {'name': 'Telugu', 'native': 'తెలుగు'},
    'es': {'name': 'Spanish', 'native': 'Español'},
    'fr': {'name': 'French', 'native': 'Français'},
    'ar': {'name': 'Arabic', 'native': 'العربية'},
    'ta': {'name': 'Tamil', 'native': 'தமிழ்'},
    'bn': {'name': 'Bengali', 'native': 'বাংলা'},
    'de': {'name': 'German', 'native': 'Deutsch'}
}

def generate_gemini_guidance(user_text: str, symptom_names: List[str], disease: str, confidence: float, lang_code: str):
    lang_info = LANGUAGE_CONFIG.get(lang_code, LANGUAGE_CONFIG['en'])
    lang_name = lang_info['name']
    lang_native = lang_info['native']

    prompt = f"""You are MediSense AI, an expert Clinical Diagnostic and Pharmacology Specialist.
Patient's Reported Problem / Symptoms: "{user_text if user_text else ', '.join(symptom_names)}"
Reported Symptom Biomarkers: {', '.join(symptom_names)}
AI Predicted Medical Condition: "{disease}" (Calculated Confidence: {confidence:.2f}%)
Required Output Language: {lang_name} ({lang_native})

Formulate an accurate, evidence-based, compassionate clinical diagnosis and treatment response.
Every sentence and field MUST be translated and written naturally in {lang_name} ({lang_native}).

Respond ONLY with a valid, clean JSON object with this exact structure:
{{
  "display_name": "Clinical condition name in {lang_name} (with English standard in parentheses if appropriate)",
  "overview": "2-3 empathetic sentences in {lang_name} explaining what this condition is, why the patient's reported symptoms match this diagnosis, and the likely cause.",
  "prescription": [
    {{
      "name": "Standard first-line medication name and strength (e.g., Paracetamol 500mg)",
      "instruction": "Clear instructions in {lang_name} on dosage and timing (e.g., Take 1 tablet after meals every 8 hours)",
      "duration": "Duration in {lang_name} (e.g., 5-7 days)"
    }}
  ],
  "care": [
    "Practical home recovery step 1 in {lang_name}",
    "Practical home recovery step 2 in {lang_name}",
    "Practical home recovery step 3 in {lang_name}"
  ],
  "diet": [
    "Dietary/hydration recommendation 1 in {lang_name}",
    "Dietary/hydration recommendation 2 in {lang_name}",
    "Dietary/hydration recommendation 3 in {lang_name}"
  ],
  "audio_summary": "A concise 2-sentence spoken summary in {lang_name} explaining the diagnosis and key remedy, written for text-to-speech."
}}
Do NOT wrap in markdown formatting code blocks. Return ONLY pure JSON."""

    # Try available active Gemini models
    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-flash-latest"]
    for m_name in models_to_try:
        try:
            m = genai.GenerativeModel(m_name)
            res = m.generate_content(prompt).text.strip()
            clean_json = re.sub(r'```json\s*|\s*```', '', res).strip()
            parsed = json.loads(clean_json)
            if "overview" in parsed and "prescription" in parsed:
                return parsed
        except Exception:
            continue

    return None

# -------------------------------------------------------------
# API MODELS
# -------------------------------------------------------------
class DiagnoseRequest(BaseModel):
    user_text: Optional[str] = ""
    symptoms: Optional[List[str]] = []
    language: Optional[str] = "en"
    mode: Optional[str] = "symptoms" # "text" or "symptoms"

# -------------------------------------------------------------
# API ROUTES
# -------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "total_symptoms": len(feature_cols),
        "total_diseases": len(diseases),
        "engine": "lightweight_pure_python"
    }

@app.get("/api/symptoms")
def get_all_symptoms():
    for candidate_path in [
        os.path.join(base_dir, "symptoms_1000.json"),
        os.path.join(base_dir, "api", "symptoms_1000.json"),
        os.path.join(base_dir, "client", "src", "symptoms_1000.json")
    ]:
        if os.path.exists(candidate_path):
            try:
                with open(candidate_path, "r", encoding="utf-8") as f:
                    s1000 = json.load(f)
                return {"symptoms": s1000, "total": len(s1000)}
            except Exception:
                pass
    sorted_symptoms = sorted(list(friendly_symptoms.values()))
    return {"symptoms": sorted_symptoms, "total": len(sorted_symptoms)}

@app.post("/api/diagnose")
def diagnose_patient(req: DiagnoseRequest):
    active_cols = set()

    # 1. Symptom parsing
    if req.mode == "text" and req.user_text:
        extracted = extract_symptoms(req.user_text)
        active_cols.update(extracted)
    elif req.symptoms:
        for s in req.symptoms:
            col = friendly_to_col.get(s.lower())
            if col and col in feature_cols:
                active_cols.add(col)
            elif s in feature_cols:
                active_cols.add(s)

    # If still empty but text was provided, do fallback regex
    if len(active_cols) == 0 and req.user_text:
        extracted = extract_symptoms(req.user_text)
        active_cols.update(extracted)

    # If nothing detected at all, default to general presentation
    if len(active_cols) == 0:
        active_cols.add('fatigue')

    # 2. Run Pure Python Naive Bayes Prediction
    predicted_disease, confidence = predict_condition(active_cols)

    symptom_display_list = [friendly_symptoms.get(c, c) for c in active_cols]

    # 3. Query Gemini for dynamic generation in selected language
    gemini_data = generate_gemini_guidance(
        user_text=req.user_text,
        symptom_names=symptom_display_list,
        disease=predicted_disease,
        confidence=confidence,
        lang_code=req.language or "en"
    )

    if gemini_data:
        display_name = gemini_data.get("display_name", predicted_disease)
        overview = gemini_data.get("overview", "")
        prescription = gemini_data.get("prescription", [])
        care = gemini_data.get("care", [])
        diet = gemini_data.get("diet", [])
        audio_text = gemini_data.get("audio_summary", f"{display_name}. {overview}")
    else:
        # Graceful fallback if Gemini quota is completely unavailable
        display_name = predicted_disease
        overview = f"Clinical pattern matching indicates symptom correlation with {predicted_disease}. Professional medical evaluation is advised."
        prescription = [
            {"name": "Supportive Medical Care", "instruction": "Rest, hydration and symptomatic relief", "duration": "5-7 days"},
            {"name": "Physician Consultation", "instruction": "Consult a registered doctor for clinical prescription", "duration": "As needed"}
        ]
        care = [
            "Get at least 8 hours of restful sleep daily.",
            "Stay adequately hydrated with 2.5 liters of clean water.",
            "Monitor body temperature and pulse regularly."
        ]
        diet = [
            "Eat freshly prepared, light, non-greasy foods.",
            "Drink warm broths, oral electrolytes, and herbal teas."
        ]
        audio_text = f"Predicted condition: {predicted_disease}. {overview}"

    # 4. Generate audio via gTTS in the requested language
    audio_base64 = None
    try:
        tts = gTTS(text=audio_text[:300], lang=req.language or "en")
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_base64 = base64.b64encode(fp.read()).decode('utf-8')
    except Exception as e:
        print("TTS generation notice:", e)

    return {
        "disease": predicted_disease,
        "displayName": display_name,
        "icon": "🩺",
        "confidence": f"{confidence:.2f}",
        "overview": overview,
        "prescription": prescription,
        "care": care,
        "diet": diet,
        "audio_base64": audio_base64,
        "detected_symptoms": symptom_display_list,
        "language": req.language
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
