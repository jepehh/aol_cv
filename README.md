﻿AOL Kelompok 15
1. Jason Patrick Winarto Hardiman - 2602163494 | jepehh
2. Michael Dimas Chrispradipta - 2602132154 | McDimas2005
3. Bintang Haidar - 2602193811 | DimmingLight

Step-by-Step Tutorial to Run the Program from GitHub Repository

1. Install Required Tools
Before starting, make sure the following tools are installed:
  a. Python (3.8 or later): Download Python, note: make sure to check the box "Add Python to PATH" during installation.
  b. Git
  c. Pip, to verify if it's installed or not, run pip --version
2. Clone the Repository
Open a terminal or command prompt.
Run the following command to clone the repository:
git clone https://github.com/jepehh/aol_cv.git
cd aol_cv

3. Set Up a Virtual Environment (Optional but Recommended)
Create a virtual environment:
python -m venv venv
venv\Scripts\activate

4. Install Dependencies
Install all required Python packages from requirements.txt:
pip install -r requirements.txt

5. Download the Required .dat Files
If the required .dat files (shape_predictor_68_face_landmarks.dat and dlib_face_recognition_resnet_model_v1.dat) do not exist, the program will dynamically download them when needed.
But there is a manual download link also which is,
  a. Shape Predictor 68 Face Landmarks:
  b. Dlib Face Recognition ResNet Model
Extract the files and place them in the project directory.

But note that this is unnecessary and is a solution if the program fails to dynamically download the .dat files.

6. Run the Program
Ensure you're in the project directory where app.py is located.
Run the Flask application:
python app.py or just press the run button.
The program will start and provide a URL (e.g., http://127.0.0.1:5000/) in the terminal. Open this URL in your browser.

7. How to Use the Application
  1. Login:
  Use the following credentials:
  Admin/Teacher: admin@test.com (Password: admin)
  Student/User: user@test.com (Password: user)
  
  2. Admin Dashboard:
  Enter the lecture topic and duration to start a lecture.
  View attendance logs, remaining time, and the teacher's IP address.
  
  3. User Dashboard:
    a. If the lecture is not ongoing, users are redirected to a waiting page.
    b. If the lecture is ongoing: Users will see their webcam feed and face detection. They can verify their attendance by matching their face and IP address.
  
  4. Access Logs and Attendance Summary
    a. Admin Dashboard: Attendance logs are displayed in real-time.
    b. Attendance Summary Page: View the attendance details after ending a session.
  
  5. Stop the Program
  To stop the program, press Ctrl + C in the terminal.
  If using a virtual environment, deactivate it by running:
  deactivate