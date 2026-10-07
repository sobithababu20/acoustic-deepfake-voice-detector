import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import tempfile
import os

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Acoustic Deepfake Detector",
    page_icon="🎙️",
    layout="centered"
)

# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #0b1020;
}

.block-container {
    max-width: 800px;
    padding-top: 50px;
    padding-bottom: 50px;
}

/* Title */

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #aab3c5;
    font-size: 17px;
    margin-bottom: 35px;
}

/* Upload */

[data-testid="stFileUploader"] {
    background: #151c2e;
    border: 1px solid #303a55;
    border-radius: 18px;
    padding: 20px;
}

/* Analyze button */

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    border: none;
    background: #4f7cff;
    color: white;
    font-size: 18px;
    font-weight: 600;
    margin-top: 15px;
}

.stButton > button:hover {
    background: #3d68df;
    color: white;
}

/* Result card */

.result-card {
    margin-top: 30px;
    padding: 35px 20px;
    border-radius: 20px;
    text-align: center;
    background: #151c2e;
}

.fake-result {
    border: 2px solid #ef4444;
}

.real-result {
    border: 2px solid #22c55e;
}

.result-title {
    font-size: 30px;
    font-weight: 700;
    color: white;
    margin-bottom: 12px;
}

.score {
    font-size: 19px;
    color: #c7cfdd;
}

.score b {
    color: white;
}

/* Bottom text */

.info {
    text-align: center;
    color: #7f8aa3;
    font-size: 14px;
    margin-top: 30px;
}

/* Hide Streamlit elements */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">🎙️ Acoustic Deepfake Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a voice recording to detect whether it is real or AI-generated.'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# MODEL
# =========================================================

MODEL_PATH = "acoustic_deepfake_balanced_model_small.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()

except Exception:
    st.error("Unable to load the detection model.")
    st.stop()

# =========================================================
# MFCC FEATURE EXTRACTION
# =========================================================

def extract_mfcc(audio_file):

    extension = os.path.splitext(audio_file.name)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension
    ) as temp_file:

        temp_file.write(audio_file.getbuffer())
        temp_path = temp_file.name

    try:

        # Load audio at 16 kHz
        audio, sr = librosa.load(
            temp_path,
            sr=16000,
            mono=True
        )

        # Extract 40 MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=16000,
            n_mfcc=40
        )

        # Fixed length = 300 frames
        max_len = 300

        if mfcc.shape[1] < max_len:

            mfcc = np.pad(
                mfcc,
                (
                    (0, 0),
                    (0, max_len - mfcc.shape[1])
                ),
                mode="constant"
            )

        else:

            mfcc = mfcc[:, :max_len]

        # Model input:
        # (40, 300)
        #      ↓
        # (40, 300, 1)
        #      ↓
        # (1, 40, 300, 1)

        mfcc = np.expand_dims(mfcc, axis=-1)
        mfcc = np.expand_dims(mfcc, axis=0)

        return mfcc

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)

# =========================================================
# AUDIO UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your audio file",
    type=["wav", "flac", "mp3"]
)

# =========================================================
# AFTER AUDIO UPLOAD
# =========================================================

if uploaded_file is not None:

    # Audio preview
    st.audio(uploaded_file)

    # Analyze button
    if st.button("🔍 Analyze Voice"):

        with st.spinner("Analyzing voice..."):

            try:

                # -----------------------------------------
                # FEATURE EXTRACTION
                # -----------------------------------------

                features = extract_mfcc(uploaded_file)

                # -----------------------------------------
                # MODEL PREDICTION
                # -----------------------------------------

                prediction = model.predict(
                    features,
                    verbose=0
                )[0][0]

                # Convert numpy value to normal float
                prediction = float(prediction)

                # -----------------------------------------
                # SPOOF
                # -----------------------------------------

                if prediction >= 0.5:

                    score = prediction * 100

                    st.markdown(
                        f'''
                        <div class="result-card fake-result">
                            <div class="result-title">
                                🔴 SPOOF / FAKE VOICE
                            </div>
                            <div class="score">
                                Detection Score:
                                <b>{score:.2f}%</b>
                            </div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )

                # -----------------------------------------
                # BONAFIDE
                # -----------------------------------------

                else:

                    score = (1 - prediction) * 100

                    st.markdown(
                        f'''
                        <div class="result-card real-result">
                            <div class="result-title">
                                🟢 BONAFIDE / REAL VOICE
                            </div>
                            <div class="score">
                                Detection Score:
                                <b>{score:.2f}%</b>
                            </div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )

            except Exception as e:

                st.error(
                    "Error while analyzing the audio."
                )

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="info">'
    'Upload an audio file and click Analyze Voice.'
    '</div>',
    unsafe_allow_html=True
)
