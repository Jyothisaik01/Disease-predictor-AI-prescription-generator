import streamlit as st
from PIL import Image

# Page config
st.set_page_config(page_title="Welcome", page_icon="👋", layout="centered", initial_sidebar_state="collapsed")

# Lottie animation support
try:
    from streamlit_lottie import st_lottie
    import requests
    def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    lottie_health = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")
except ImportError:
    st.warning("Install streamlit-lottie for enhanced animations: pip install streamlit-lottie")

# Centered container
st.markdown("""
    <style>
    .center {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 20px;
    }
    .animation-space {
        margin-bottom: 0px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #0072ff 60%, #00c6ff 100%);
        color: #fff;
        font-size: 1.25rem;
        font-weight: 700;
        padding: 16px 44px;
        border-radius: 32px;
        margin-top: 28px;
        border: none;
        box-shadow: 0 4px 24px rgba(0,114,255,0.18);
        cursor: pointer;
        transition: background 0.2s, transform 0.18s, box-shadow 0.2s;
        outline: none;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #00c6ff 60%, #0072ff 100%);
        color: #fff;
        transform: scale(1.06) translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,114,255,0.28);
        animation: bounce 0.38s 1;
    }
    @keyframes bounce {
        0% { transform: scale(1.06) translateY(-2px); }
        30% { transform: scale(1.10) translateY(-8px); }
        60% { transform: scale(0.98) translateY(2px); }
        100% { transform: scale(1.06) translateY(-2px); }
    }
    .animated-title {
        font-size: 2.8rem;
        font-weight: bold;
        color: #0072ff;
        text-shadow: 2px 2px 10px rgba(179,224,255,0.35), 0 2px 8px rgba(0,91,181,0.18);
        animation: pop 1.2s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }
    .animated-subheader {
        font-size: 1.5rem;
        # color: #222;
        text-shadow: 1px 1px 8px rgba(204,230,255,0.28);
        animation: fadein 2s;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
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
    </style>
""", unsafe_allow_html=True)

# Content
st.markdown('<div class="center">', unsafe_allow_html=True)
try:
    st.markdown('<div class="animation-space">', unsafe_allow_html=True)
    st_lottie(lottie_health, height=220, key="health")
    st.markdown('</div>', unsafe_allow_html=True)
except ImportError:
    pass

st.markdown('<div class="animated-title">🩺 Disease Prediction Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="animated-subheader">🤖 Welcome to your AI-powered health advisor.</div>', unsafe_allow_html=True)
st.markdown('<div style="margin-bottom:12px;">Click below to get started with symptom analysis and receive medical guidance.</div>', unsafe_allow_html=True)

# Redirect button
start_btn = st.button("🚀 Get Started", key="get_started_btn")
if start_btn:
    st.balloons()
    st.switch_page("pages/app.py")

st.markdown("</div>", unsafe_allow_html=True)
