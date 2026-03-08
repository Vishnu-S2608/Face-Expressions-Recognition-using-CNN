# EmoSense AI – Facial Emotion Recognition using Deep Learning

EmoSense AI is a deep learning-based web application that detects **human emotions from facial images**.
The system uses a **Convolutional Neural Network (CNN)** trained on a facial expression dataset to classify emotions from a face image.

The application is built using:

* **TensorFlow / Keras** for deep learning
* **OpenCV** for face detection
* **Streamlit** for the interactive web interface

Users can simply upload an image, and the model will automatically detect faces and **predict the emotion expressed in the image**.

---

# Emotion Recognition

When an image is uploaded, the system detects faces and classifies them into one of the following **seven emotions**:

* Angry 😠
* Disgust 🤢
* Fear 😨
* Happy 😄
* Neutral 😐
* Sad 😢
* Surprise 😲

The model also displays a **confidence score** and a **probability distribution of all emotions**.

---

# Project Workflow

The application follows this workflow:

1. User uploads an image through the Streamlit interface.
2. The image is converted into a NumPy array.
3. The image is converted to **grayscale**.
4. **OpenCV Haar Cascade Classifier** detects faces in the image.
5. Each detected face is cropped and resized to **48 × 48 pixels**.
6. The processed face image is normalized and passed to the **CNN model**.
7. The trained model predicts the **emotion probabilities**.
8. The predicted emotion and confidence score are displayed on the UI.

---

# CNN Architecture

The model is a **Convolutional Neural Network (CNN)** designed for facial expression recognition.

### Architecture Overview

Input Image
48 × 48 grayscale

Layers:

1. Convolution Layer (64 filters)

2. Batch Normalization

3. ReLU Activation

4. Max Pooling

5. Dropout

6. Convolution Layer (128 filters)

7. Batch Normalization

8. ReLU Activation

9. Max Pooling

10. Dropout

11. Convolution Layer (512 filters)

12. Batch Normalization

13. ReLU Activation

14. Max Pooling

15. Dropout

16. Convolution Layer (512 filters)

17. Batch Normalization

18. ReLU Activation

19. Max Pooling

20. Dropout

21. Flatten Layer

Fully Connected Layers:

* Dense (256)
* Dense (512)
* Output Dense (7 classes with Softmax)

The model is trained using the **Adam optimizer** with **categorical cross-entropy loss**.

---

# Model File – `model.h5`

The file `model.h5` contains the **trained CNN model**.

This file stores:

* Model architecture
* Trained weights
* Optimizer configuration

During prediction, the Streamlit application loads this model and uses it to **predict emotions from face images**.

---

# Face Detection – Haar Cascade XML

The project uses an OpenCV Haar Cascade classifier for detecting faces.

File used:

`HaarcascadeclassifierCascadeClassifier.xml`

This XML file contains a **pre-trained face detection model** which allows the system to:

1. Detect faces in an image
2. Extract face regions
3. Send the detected face to the CNN model for emotion classification

---

# UI Design (Streamlit)

The user interface is built using **Streamlit** and includes several visual features:

* Modern gradient background
* Glassmorphism UI cards
* Animated UI elements
* Emotion emoji indicators
* Emotion confidence bar
* Emotion probability distribution chart

Features of the interface:

* Upload image option
* Automatic face detection
* Real-time emotion prediction
* Emotion probability visualization

---

# How to Run the Project

Follow these steps to run the project on your system.

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
```

Then navigate to the project folder:

```bash
cd YOUR_REPOSITORY_NAME
```

---

## Step 2 — Create Virtual Environment

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Mac / Linux

```bash
source venv/bin/activate
```

---

## Step 3 — Install Dependencies

First install **TensorFlow** separately:

```bash
pip install tensorflow==2.15.0
```

Then install the remaining packages:

```bash
pip install streamlit opencv-python keras pillow numpy matplotlib
```

---

## Step 4 — Run the Application

Start the Streamlit application:

```bash
streamlit run main.py
```

Streamlit will launch the app in your browser at:

```
http://localhost:8501
```

Upload an image and the system will detect the **emotion expressed in the face**.

---

# Project Structure

```
EmoSense-AI
│
├── main.py
├── emotion_recognition_cnn.py
├── model.h5
├── HaarcascadeclassifierCascadeClassifier.xml
├── README.md
└── requirements.txt
```

---

# Future Improvements

Possible improvements for the project:

* Real-time **webcam emotion detection**
* Improve CNN accuracy using **Transfer Learning**
* Deploy the application online using **Streamlit Cloud**
* Replace Haar Cascade with **Deep Learning face detection**
* Add **multi-face emotion tracking**
* Train with a larger dataset for better performance
* Add **emotion analytics dashboard**

---

# Dataset

The model is trained on the **Face Expression Recognition Dataset**.

Dataset link:

https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

---

# Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Matplotlib
* Streamlit

---

# License

This project is intended for **educational and research purposes**.
