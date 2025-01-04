from flask import Flask, render_template, redirect, url_for, request, session, jsonify
from datetime import datetime, timedelta
import cv2
import dlib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import socket
import os
import requests

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Global Variables
ongoing_lecture = {"teacher_ip": None, "end_time": None, "topic": None}
attendance_log = []
verification_status = {"verified": False, "name": None, "confidence": None}  # Tracks live verification status

# File download function
def download_file(url, file_path):
    if not os.path.exists(file_path):
        print(f"Downloading {file_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"{file_path} downloaded.")

# URLs for model files (adjusted for direct download)
shape_predictor_url = "https://drive.google.com/uc?id=18w8-gkA_GXk3nn2jki5eyomuALlKSzgE&export=download"
dlib_resnet_url = "https://drive.google.com/uc?id=1rZMu8kb4hm7JnmG93yE1c30XtQrGpCie&export=download"

# File paths
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
dlib_resnet_path = "dlib_face_recognition_resnet_model_v1.dat"

# Download required files if they don't exist
download_file(shape_predictor_url, shape_predictor_path)
download_file(dlib_resnet_url, dlib_resnet_path)

# Load Face Recognition Model
model_save_path = "neural_net_model.h5"
encoder_save_path = "classes.npy"

model = tf.keras.models.load_model(model_save_path)
label_encoder = np.load(encoder_save_path, allow_pickle=True)

# Load dlib models
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_rec_model = dlib.face_recognition_model_v1(dlib_resnet_path)

def get_wifi_ip():
    """Get the device's Wi-Fi IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"  # Fallback to localhost

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email == 'admin@test.com' and password == 'admin':
            session['user'] = 'admin'
            return redirect(url_for('admin_dashboard'))
        elif email == 'user@test.com' and password == 'user':
            session['user'] = 'user'
            return redirect(url_for('user_dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin_dashboard():
    if 'user' not in session or session['user'] != 'admin':
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'start_lecture' in request.form:
            topic = request.form['topic']
            duration = int(request.form['duration'])
            end_time = datetime.now() + timedelta(minutes=duration)
            teacher_ip = get_wifi_ip()  # Get Wi-Fi IP address
            ongoing_lecture.update({
                "teacher_ip": teacher_ip,
                "end_time": end_time,
                "topic": topic
            })
        elif 'end_lecture' in request.form:
            ongoing_lecture.update({"teacher_ip": None, "end_time": None, "topic": None})
            attendance_log.clear()

    remaining_time = calculate_remaining_time()
    return render_template('admin_dashboard.html', lecture=ongoing_lecture, remaining_time=remaining_time, attendance_log=attendance_log)

@app.route('/user', methods=['GET'])
def user_dashboard():
    if 'user' not in session or session['user'] != 'user':
        return redirect(url_for('login'))
    if ongoing_lecture["end_time"] is None:
        return redirect(url_for('waiting'))
    return render_template('user_dashboard.html', topic=ongoing_lecture["topic"])

@app.route('/waiting', methods=['GET'])
def waiting():
    if 'user' not in session or session['user'] != 'user':
        return redirect(url_for('login'))
    if ongoing_lecture["end_time"]:
        return redirect(url_for('user_dashboard'))
    return render_template('waiting.html')

@app.route('/remaining_time')
def remaining_time():
    """Return remaining time for the lecture."""
    remaining = calculate_remaining_time()
    if remaining is None:
        return jsonify({'time': None})
    return jsonify({'time': str(remaining)})

@app.route('/live_verify', methods=['POST'])
def live_verify():
    """Handle real-time face recognition to update verification status."""
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Detect face
    faces = face_detector(image, 1)
    if not faces:
        verification_status["verified"] = False
        verification_status["name"] = None
        verification_status["confidence"] = None
        return jsonify({'status': 'No face detected', 'verified': verification_status["verified"]})

    for rect in faces:
        shape = shape_predictor(image, rect)
        embedding = np.array(face_rec_model.compute_face_descriptor(image, shape)).reshape(1, -1)
        preds = model.predict(embedding)
        confidence = np.max(preds)
        label = label_encoder[np.argmax(preds)]

        if confidence >= 0.5:
            verification_status["verified"] = True
            verification_status["name"] = label
            verification_status["confidence"] = round(confidence * 100, 2)
        else:
            verification_status["verified"] = False
            verification_status["name"] = None
            verification_status["confidence"] = None

        return jsonify({
            'status': 'Face recognized' if verification_status["verified"] else 'Face not recognized',
            'name': label if verification_status["verified"] else None,
            'confidence': round(confidence * 100, 2) if verification_status["verified"] else None,
            'verified': verification_status["verified"],
            'rect': [rect.left(), rect.top(), rect.right(), rect.bottom()]
        })

    verification_status["verified"] = False
    verification_status["name"] = None
    verification_status["confidence"] = None
    return jsonify({'status': 'No face detected', 'verified': verification_status["verified"]})

@app.route('/verify_ip', methods=['POST'])
def verify_ip():
    """Verify if the user's IP matches the lecturer's IP and log attendance."""
    if 'user' not in session or session['user'] != 'user':
        return jsonify({'status': 'Unauthorized', 'user_ip': get_wifi_ip()}), 401

    user_ip = get_wifi_ip()  # Get user's Wi-Fi IP
    teacher_ip = ongoing_lecture.get("teacher_ip")

    if not teacher_ip:
        return jsonify({'status': 'No ongoing lecture', 'user_ip': user_ip}), 400

    if verification_status["verified"]:  # Check if face is verified
        if user_ip.split('.')[0:3] == teacher_ip.split('.')[0:3]:  # Compare subnet
            attendance_log.append({
                "name": verification_status["name"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "confidence": verification_status["confidence"],
                "ip": user_ip
            })
            return jsonify({'status': 'Attendance logged successfully', 'user_ip': user_ip})
        return jsonify({'status': "Your IP doesn't match the Classroom's IP", 'user_ip': user_ip})
    return jsonify({'status': 'Face not verified. Please face the camera.', 'user_ip': user_ip})

def calculate_remaining_time():
    """Calculate remaining time for the lecture."""
    if not ongoing_lecture["end_time"]:
        return None
    remaining = ongoing_lecture["end_time"] - datetime.now()
    return max(remaining, timedelta(seconds=0))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
