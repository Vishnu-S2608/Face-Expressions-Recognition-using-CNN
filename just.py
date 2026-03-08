import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# -------------------- Load Model and Classifier --------------------
MODEL_PATH = "model.h5"
FACE_CASCADE_PATH = "HaarcascadeclassifierCascadeClassifier.xml"

st.set_page_config(page_title="Emotion Recognition", layout="centered")

@st.cache_resource
def load_emotion_model():
    model = load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_face_detector():
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    return face_cascade

model = load_emotion_model()
face_cascade = load_face_detector()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# -------------------- Streamlit UI --------------------
st.title("😊 Emotion Recognition from Image")
st.write("Upload an image and the model will predict the detected emotions.")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to OpenCV format
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.warning("⚠️ No face detected. Try uploading a clearer face image.")
    else:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            # Preprocessing (match training setup)
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Prediction
            preds = model.predict(roi)[0]
            label = emotion_labels[preds.argmax()]
            confidence = preds[preds.argmax()] * 100

            # Draw rectangle and label on image
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img_array, f"{label} ({confidence:.2f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display results
            st.subheader(f"Predicted Emotion: {label}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.bar_chart(dict(zip(emotion_labels, preds)))

        # Show annotated image
        st.image(img_array, caption="Detected Faces and Emotions", use_container_width=True)

st.markdown("---")
st.caption("Model: Custom CNN trained on FER dataset | Built with Streamlit, Keras, OpenCV")
