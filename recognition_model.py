import cv2
import dlib
import numpy as np
import tensorflow as tf

# Paths
model_save_path = "neural_net_model.h5"
encoder_save_path = "classes.npy"

# Load the trained neural network model and label encoder
model = tf.keras.models.load_model(model_save_path)
label_encoder = np.load(encoder_save_path, allow_pickle=True)

# Load dlib models
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Face Recognition Function
def recognize_face(image_file):
    # Read image as an array
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)

    if len(faces) == 0:
        return 0, False  # No face detected

    # Process the first detected face
    shape = shape_predictor(image, faces[0])
    face_embedding = np.array(face_rec_model.compute_face_descriptor(image, shape)).reshape(1, -1)

    # Predict label using the trained model
    preds = model.predict(face_embedding)
    confidence = np.max(preds)
    label = label_encoder[np.argmax(preds)]

    recognized = confidence >= 0.6  # Threshold for recognition
    return confidence * 100, recognized
