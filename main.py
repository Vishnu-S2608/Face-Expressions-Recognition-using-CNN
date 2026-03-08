import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# -------------------- Load Model and Classifier --------------------
MODEL_PATH = "model.h5"
FACE_CASCADE_PATH = "HaarcascadeclassifierCascadeClassifier.xml"

st.set_page_config(
    page_title="EmoSense AI",
    layout="centered",
    page_icon="🎭"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>

/* ═══════════════════════════════════════════
   GLOBAL RESET & BACKGROUND
═══════════════════════════════════════════ */
:root {
    --pink:    #FF6B9D;
    --purple:  #A855F7;
    --blue:    #3B82F6;
    --cyan:    #06B6D4;
    --orange:  #F97316;
    --yellow:  #FBBF24;
    --green:   #10B981;
    --white:   #FFFFFF;
    --glass-bg:   rgba(255, 255, 255, 0.25);
    --glass-border: rgba(255, 255, 255, 0.45);
    --glass-shadow: 0 8px 32px rgba(99, 50, 200, 0.18);
    --text-dark:  #1E1B4B;
    --text-mid:   #4C4582;
    --text-light: rgba(30, 27, 75, 0.55);
}

/* Full vivid gradient background */
.stApp {
    background: linear-gradient(135deg,
        #D4EDDA 0%,
        #C8E6C9 20%,
        #DCEDC8 40%,
        #E8F5E9 60%,
        #F1F8E9 80%,
        #D0E8D0 100%) !important;
    min-height: 100vh;
}

/* Animated floating blobs */
.stApp::before {
    content: '';
    position: fixed;
    top: -120px; left: -120px;
    width: 520px; height: 520px;
    background: radial-gradient(circle, rgba(168,85,247,0.22) 0%, transparent 70%);
    border-radius: 50%;
    animation: blob1 8s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: -100px; right: -100px;
    width: 480px; height: 480px;
    background: radial-gradient(circle, rgba(6,182,212,0.20) 0%, transparent 70%);
    border-radius: 50%;
    animation: blob2 10s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
@keyframes blob1 {
    0%,100% { transform: translate(0,0) scale(1); }
    50%      { transform: translate(60px,40px) scale(1.1); }
}
@keyframes blob2 {
    0%,100% { transform: translate(0,0) scale(1); }
    50%      { transform: translate(-40px,-50px) scale(1.08); }
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* ═══════════════════════════════════════════
   LAYOUT
═══════════════════════════════════════════ */
.main .block-container {
    max-width: 700px !important;
    padding: 2rem 1.5rem 4rem !important;
    position: relative;
    z-index: 1;
}

/* ═══════════════════════════════════════════
   HERO
═══════════════════════════════════════════ */
.hero {
    text-align: center;
    padding: 3rem 2rem 2.5rem;
    background: rgba(255,255,255,0.28);
    backdrop-filter: blur(24px) saturate(200%);
    -webkit-backdrop-filter: blur(24px) saturate(200%);
    border: 1.5px solid rgba(255,255,255,0.55);
    border-radius: 28px;
    box-shadow: 0 12px 40px rgba(120,80,220,0.14), inset 0 1px 0 rgba(255,255,255,0.75);
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--pink), var(--purple), var(--blue), var(--cyan));
    border-radius: 28px 28px 0 0;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(168,85,247,0.15), rgba(59,130,246,0.15));
    border: 1px solid rgba(168,85,247,0.30);
    border-radius: 99px;
    padding: 6px 16px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--purple);
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: clamp(2.4rem, 5vw, 3.6rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 0.9rem;
    background: linear-gradient(135deg, var(--pink) 0%, var(--purple) 40%, var(--blue) 80%, var(--cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1rem;
    font-weight: 400;
    color: var(--text-mid);
    line-height: 1.65;
    max-width: 420px;
    margin: 0 auto;
}
.hero-pills {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 1.6rem;
}
.pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 13px;
    border-radius: 99px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    border: 1px solid;
}
.pill-pink   { background:rgba(255,107,157,0.12); border-color:rgba(255,107,157,0.35); color:#C2185B; }
.pill-purple { background:rgba(168,85,247,0.12);  border-color:rgba(168,85,247,0.35);  color:#7C3AED; }
.pill-cyan   { background:rgba(6,182,212,0.12);   border-color:rgba(6,182,212,0.35);   color:#0E7490; }

/* ═══════════════════════════════════════════
   SECTION LABELS
═══════════════════════════════════════════ */
.sec-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--purple);
    margin: 2rem 0 0.65rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(168,85,247,0.35), transparent);
    border-radius: 99px;
}

/* ═══════════════════════════════════════════
   FILE UPLOADER
═══════════════════════════════════════════ */
.upload-wrap {
    background: rgba(255,255,255,0.28);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 2px dashed rgba(168,85,247,0.40);
    border-radius: 24px;
    padding: 2rem 1.5rem;
    box-shadow: 0 8px 30px rgba(120,80,220,0.10), inset 0 1px 0 rgba(255,255,255,0.7);
    margin-bottom: 1.5rem;
}

[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
    border: none !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 0.5rem 0 !important;
}
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] small {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: black !important;
    font-size: 0.88rem !important;
    text-align: center !important;
}
[data-testid="stFileUploadDropzone"] button {
    background: linear-gradient(135deg, var(--purple), var(--blue)) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.65rem 2.2rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    cursor: pointer !important;
    box-shadow: 0 4px 18px rgba(168,85,247,0.40) !important;
    margin-top: 6px !important;
}

/* ═══════════════════════════════════════════
   IMAGE DISPLAY
═══════════════════════════════════════════ */
[data-testid="stImage"] {
    border-radius: 20px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 32px rgba(120,80,220,0.14) !important;
    border: 1.5px solid rgba(255,255,255,0.60) !important;
}
[data-testid="stImage"] img { border-radius: 20px !important; }
[data-testid="stCaptionContainer"] p {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-light) !important;
    font-size: 0.75rem !important;
    text-align: center !important;
    margin-top: 0.4rem !important;
}

/* ═══════════════════════════════════════════
   RESULT CARD
═══════════════════════════════════════════ */
.result-card {
    background: rgba(255,255,255,0.35);
    backdrop-filter: blur(24px) saturate(200%);
    -webkit-backdrop-filter: blur(24px) saturate(200%);
    border: 1.5px solid rgba(255,255,255,0.60);
    border-radius: 24px;
    padding: 1.8rem 2rem;
    margin: 0.5rem 0 1.2rem;
    box-shadow: 0 10px 40px rgba(120,80,220,0.13), inset 0 1px 0 rgba(255,255,255,0.8);
    display: flex;
    align-items: center;
    gap: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--pink), var(--purple), var(--blue));
}
.result-emoji-box {
    width: 76px; height: 76px;
    flex-shrink: 0;
    background: linear-gradient(135deg, rgba(168,85,247,0.15), rgba(59,130,246,0.15));
    border: 1.5px solid rgba(168,85,247,0.25);
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.5rem;
    box-shadow: 0 4px 20px rgba(168,85,247,0.18);
}
.result-info { flex: 1; min-width: 0; }
.result-top-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--purple);
    margin-bottom: 0.3rem;
}
.result-emotion-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, var(--pink), var(--purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.6rem;
}
.conf-row {
    display: flex;
    align-items: center;
    gap: 10px;
}
.conf-track {
    flex: 1;
    height: 6px;
    background: rgba(168,85,247,0.12);
    border-radius: 99px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--pink), var(--purple), var(--blue));
    border-radius: 99px;
    box-shadow: 0 0 10px rgba(168,85,247,0.5);
}
.conf-pct {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--purple);
    white-space: nowrap;
    min-width: 48px;
    text-align: right;
}

/* ═══════════════════════════════════════════
   BAR CHART
═══════════════════════════════════════════ */
[data-testid="stVegaLiteChart"],
[data-testid="stBarChart"] {
    background: rgba(255,255,255,0.30) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1.5px solid rgba(255,255,255,0.55) !important;
    border-radius: 20px !important;
    padding: 1.2rem 1rem 0.8rem !important;
    box-shadow: 0 6px 24px rgba(120,80,220,0.10) !important;
}

/* ═══════════════════════════════════════════
   WARNINGS
═══════════════════════════════════════════ */
[data-testid="stAlert"] {
    background: rgba(249,115,22,0.10) !important;
    border: 1.5px solid rgba(249,115,22,0.35) !important;
    border-radius: 14px !important;
}
[data-testid="stAlert"] p {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #C2410C !important;
}

/* ═══════════════════════════════════════════
   FOOTER
═══════════════════════════════════════════ */
.footer-card {
    text-align: center;
    margin-top: 3rem;
    padding: 1.2rem;
    background: rgba(255,255,255,0.22);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.45);
    border-radius: 16px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    color: var(--text-light);
}
.footer-card b { color: var(--purple); }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(var(--pink), var(--purple));
    border-radius: 99px;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load resources
# ──────────────────────────────────────────────
@st.cache_resource
def load_emotion_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(FACE_CASCADE_PATH)

model        = load_emotion_model()
face_cascade = load_face_detector()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_emoji  = {
    'Angry':'😠','Disgust':'🤢','Fear':'😨',
    'Happy':'😄','Neutral':'😐','Sad':'😢','Surprise':'😲'
}

# ──────────────────────────────────────────────
# Hero
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ &nbsp; AI Powered &nbsp; ✦</div>
    <div class="hero-title">EmoSense AI</div>
    <div class="hero-sub">
        Upload a portrait and our neural network will instantly
        decode the emotions written across every face.
    </div>
    <div class="hero-pills">
        <span class="pill pill-pink">😊 7 Emotions</span>
        <span class="pill pill-purple">⚡ Real-time</span>
        <span class="pill pill-cyan">🧠 CNN Model</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Upload
# ──────────────────────────────────────────────
st.markdown('<div class="sec-label">Upload Image</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-wrap">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    label="upload",
    type=["jpg","jpeg","png"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="sec-label">Input Image</div>', unsafe_allow_html=True)
    st.image(image, caption="Uploaded portrait", use_container_width=True)

    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.warning("⚠️  No face detected — please try a clearer, front-facing portrait.")
    else:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds      = model.predict(roi)[0]
            label      = emotion_labels[preds.argmax()]
            confidence = preds[preds.argmax()] * 100
            emoji      = emotion_emoji.get(label, "🎭")

            cv2.rectangle(img_array, (x, y), (x+w, y+h), (168, 85, 247), 2)
            cv2.putText(
                img_array,
                f"{label}  {confidence:.1f}%",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.82,
                (168, 85, 247), 2
            )

            st.markdown('<div class="sec-label">Detected Emotion</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-card">
                <div class="result-emoji-box">{emoji}</div>
                <div class="result-info">
                    <div class="result-top-label">Predicted Emotion</div>
                    <div class="result-emotion-name">{label}</div>
                    <div class="conf-row">
                        <div class="conf-track">
                            <div class="conf-fill" style="width:{confidence:.1f}%"></div>
                        </div>
                        <div class="conf-pct">{confidence:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="sec-label">Emotion Distribution</div>', unsafe_allow_html=True)
            st.bar_chart(dict(zip(emotion_labels, preds)))

        st.markdown('<div class="sec-label">Detection Output</div>', unsafe_allow_html=True)
        st.image(img_array, caption="Faces detected with emotion overlay", use_container_width=True)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("""
<div class="footer-card">
    <b>EmoSense AI</b> &nbsp;·&nbsp; Custom CNN trained on FER Dataset &nbsp;·&nbsp; Built with Streamlit, Keras & OpenCV
</div>
""", unsafe_allow_html=True)