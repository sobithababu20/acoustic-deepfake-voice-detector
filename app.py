import streamlit as st
import librosa
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Acoustic Deepfake Detector", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
.stApp { background: #f5f7fb; }
.hero { background: linear-gradient(135deg,#111827,#1d4ed8); padding:40px; border-radius:20px; text-align:center; color:white; margin-bottom:30px; }
.hero h1 { font-size:42px; margin-bottom:8px; }
.hero p { font-size:18px; color:#dbeafe; }
.card { background:white; padding:25px; border-radius:18px; margin-bottom:20px; box-shadow:0 4px 18px rgba(0,0,0,.08); }
.title { font-size:24px; font-weight:700; margin-bottom:15px; color:#111827; }
.result-fake { background:#fff1f2; border:2px solid #ef4444; padding:30px; border-radius:18px; text-align:center; }
.result-real { background:#ecfdf5; border:2px solid #10b981; padding:30px; border-radius:18px; text-align:center; }
.result { font-size:34px; font-weight:800; }
.score { font-size:20px; margin-top:10px; }
.footer { text-align:center; color:#6b7280; padding:30px; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "quick_acoustic_deepfake_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

def extract_mfcc(audio_file):
    audio, sr = librosa.load(audio_file, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    max_len = 300
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0,max_len-mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]
    return np.expand_dims(np.expand_dims(mfcc, axis=0), axis=-1)

st.markdown("""
<div class="hero">
<h1>🎙️ Acoustic Deepfake Detector</h1>
<p>AI-Powered Voice Authenticity Analysis using MFCC and CNN</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<div class="title">🔍 Detect Fake or AI-Generated Voices</div>
<p>Upload an audio file and the deep learning model will analyze its acoustic characteristics and classify the speech as <b>Bonafide</b> or <b>Spoof</b>.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card"><div class="title">📁 Upload Audio</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav","flac","mp3"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <div class="title">🧠 Detection Pipeline</div>
    <p>🎵 Audio Input</p><p>↓</p><p>🔊 Preprocessing</p><p>↓</p>
    <p>📊 MFCC Feature Extraction</p><p>↓</p><p>🧠 CNN Classification</p><p>↓</p>
    <p>✅ Bonafide / ❌ Spoof</p>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file:
    st.markdown('<div class="card"><div class="title">🔊 Audio Preview</div>', unsafe_allow_html=True)
    st.audio(uploaded_file)
    st.markdown("</div>", unsafe_allow_html=True)

    try:
        features = extract_mfcc(uploaded_file)
        prediction = model.predict(features, verbose=0)[0][0]

        if prediction >= 0.5:
            confidence = prediction * 100
            st.markdown(f"""
            <div class="result-fake">
            <div class="result">❌ SPOOF / FAKE VOICE</div>
            <div class="score">Detection Score: <b>{confidence:.2f}%</b></div>
            <p>The model classified this audio as spoofed or synthetic speech.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence = (1-prediction) * 100
            st.markdown(f"""
            <div class="result-real">
            <div class="result">✅ BONAFIDE / REAL VOICE</div>
            <div class="score">Detection Score: <b>{confidence:.2f}%</b></div>
            <p>The model classified this audio as bonafide speech.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<br><div class="card"><div class="title">📊 Analysis Details</div></div>', unsafe_allow_html=True)
        a,b,c = st.columns(3)
        a.metric("Sampling Rate","16 kHz")
        b.metric("MFCC Features","40")
        c.metric("Classifier","CNN")
    except Exception:
        st.error("Unable to process this audio file.")

st.markdown("""
<div class="card">
<div class="title">📌 About the Project</div>
<b>Dataset:</b> ASVspoof 2019 Logical Access (LA)<br><br>
<b>Feature Extraction:</b> MFCC<br><br>
<b>Deep Learning:</b> Convolutional Neural Network (CNN)<br><br>
<b>Classification:</b> Bonafide vs Spoof<br><br>
<b>Purpose:</b> Detection of AI-generated and manipulated speech
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="footer"><b>Acoustic Deepfake and Voice Cloning Discriminator</b><br>Machine Learning PBL Project</div>', unsafe_allow_html=True)
