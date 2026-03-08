# Face-Expression-Recognition-using-Deep-Learning
This project implements a convolutional neural network (CNN) to recognize facial expressions of seven different emotions: angry, disgust, fear, happy, neutral, sad, and surprise. The model is trained on the Face expression recognition dataset. Dataset E-link: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset.


Requirements:-
Python 3.11,
keras~=2.12.0rc0,
tensorflow,
numpy~=1.23.5,
matplotlib~=3.7.0,
pandas~=1.5.3,
seaborn,
opencv-contrib-python==4.7.0.68


Installation:-
1. First install the python.
2. After installing python. Install the packages listed in the requirements.txt. Use the command pip install -r requirements.txt 


Usage:-
1. Clone or download the repository.
2. Install the requirements
2. Run the main.py script using the following command:
   python main.py
3. The script will launch the webcam and start detecting emotions in real-time.


Files:-
1. main.py: This file is the entry point of the application. It loads the trained model and uses it to predict the emotions of faces in real-time using a webcam.
2. emotion_recognition_cnn.py: This file contains the Python code for building and training the CNN.
3. HaarcascadeclassifierCascadeClassifier.xml: The pre-trained Haar Cascade Classifier for detecting faces in images.
4. model.h5: The pre-trained Keras model for emotion detection.


"""
This code is an implementation of a deep learning model for face expression recognition using the Keras
framework. The goal of the model is to classify facial expressions into seven categories:
angry, disgust, fear, happy, neutral, sad, and surprise.

The code first imports the necessary libraries including Keras, matplotlib, and os.
It then displays a sample of images of the expression "disgust" from the dataset to visualize the data.

The training and validation data are created using Keras' ImageDataGenerator which reads images from the
given directory and returns batches of images and labels. The train_set and test_set are initialized with
the parameters such as target size, batch size, and color mode.

The model is built using a sequential model in Keras, with a series of Convolutional Neural Network (CNN)
layers followed by fully connected layers. The first four layers are CNN layers, each followed by batch
normalization, activation, max pooling, and dropout. The output of the last CNN layer is then flattened and
fed into two fully connected layers, each with a batch normalization, activation, dropout, and dense layer.
The last dense layer has a softmax activation function which returns the probabilities for each class.

The model is then compiled using the Adam optimizer, categorical cross-entropy as the loss function,
and accuracy as the metric to evaluate the performance.

The model is fitted using the fit_generator() function in Keras. The function takes the training and validation
data, the number of epochs, and a list of callback functions that monitor the training process, and then
updates the weights in the model accordingly. In this code, three callback functions are used: ModelCheckpoint,
EarlyStopping, and ReduceLROnPlateau.

Finally, the code visualizes the loss and accuracy of the model during the training process using the
matplotlib library. The history object returned by the fit_generator() function is used to plot the
loss and accuracy curves for both training and validation data.

"""


libraries

pip install streamlit opencv-python tensorflow keras pillow numpy