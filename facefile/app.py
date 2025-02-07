from flask import Flask, Response, render_template, jsonify
import cv2
import numpy as np
import time
import threading

app = Flask(__name__)

# Load face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Load eye cascade
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# Load smile cascade
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Add a global variable to store latest analysis
latest_analysis = {
    'age': 'Unknown',
    'gender': 'Unknown',
    'emotion': 'Unknown',
    'health': 'Unknown',
    'distance': 'N/A',
    'fps': 'N/A',
    'objects': []
}

def detect_emotion(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Detect smile
    smiles = smile_cascade.detectMultiScale(gray, 1.7, 20)
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Basic emotion analysis
    if len(smiles) > 0:
        return "Happy"
    elif len(eyes) >= 2:
        # Check eye openness using average intensity
        eye_intensity = np.mean(gray[eyes[0][1]:eyes[0][1]+eyes[0][3], eyes[0][0]:eyes[0][0]+eyes[0][2]])
        if eye_intensity < 100:
            return "Tired"
        else:
            return "Neutral"
    return "Unknown"

def analyze_face_features(face_img):
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Detect facial features
    eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 4)
    smiles = smile_cascade.detectMultiScale(gray_face, 1.7, 20)
    
    # Improved age estimation
    height, width = face_img.shape[:2]
    face_size = height * width
    texture_value = np.std(gray_face)
    
    # More precise age ranges
    if face_size < 10000:
        estimated_age = "Child (0-12)"
    elif face_size < 15000:
        estimated_age = "Teen (13-19)"
    elif face_size < 20000:
        if texture_value < 30:
            estimated_age = "Young Adult (20-30)"
        else:
            estimated_age = "Adult (31-50)"
    else:
        if texture_value > 40:
            estimated_age = "Senior (50+)"
        else:
            estimated_age = "Adult (31-50)"
    
    # Improved gender estimation
    ratio = width / height
    brightness = np.mean(gray_face)
    texture = np.std(gray_face)
    
    # More sophisticated gender detection
    gender_score = 0
    gender_score += 1 if ratio < 0.95 else -1
    gender_score += 1 if brightness > 120 else -1
    gender_score += 1 if texture < 35 else -1
    
    estimated_gender = "Female" if gender_score > 0 else "Male"
    
    return {
        'age': estimated_age,
        'gender': estimated_gender,
        'emotion': detect_emotion(face_img),
        'eyes_detected': len(eyes),
        'smiling': len(smiles) > 0
    }

def analyze_face_health(face_img):
    # Convert to different color spaces for better analysis
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
    
    # Get various color metrics
    avg_saturation = np.mean(hsv[:, :, 1])
    avg_value = np.mean(hsv[:, :, 2])
    avg_cr = np.mean(ycrcb[:, :, 1])  # Red difference
    avg_cb = np.mean(ycrcb[:, :, 2])  # Blue difference
    
    # Complex health analysis
    if avg_value < 100:
        return "Tired"
    elif avg_saturation < 50 and avg_value > 150:
        return "Pale"
    elif avg_cr > 150:  # High red component
        return "Flushed"
    elif 140 > avg_cr > 135 and 120 > avg_cb > 110:
        return "Healthy"
    else:
        return "Check Health"

def generate_frames():
    camera = cv2.VideoCapture(0)
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # Calculate FPS
        fps_counter += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
            
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        # Update global analysis with default values if no face detected
        if len(faces) == 0:
            latest_analysis.update({
                'age': 'No face detected',
                'gender': 'No face detected',
                'emotion': 'No face detected',
                'health': 'No face detected',
                'distance': 'N/A',
                'fps': str(fps),
                'objects': []
            })
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            
            try:
                # Analyze face
                features = analyze_face_features(face_img)
                health_status = analyze_face_health(face_img)
                distance = round(1 / (w / 100), 2)
                
                # Update global analysis
                latest_analysis.update({
                    'age': features['age'],
                    'gender': features['gender'],
                    'emotion': features['emotion'],
                    'health': health_status,
                    'distance': f"{distance}m",
                    'fps': str(fps),
                    'objects': []
                })
                
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Draw labels
                info_text = f"{features['gender']}, {features['age']}"
                emotion_text = f"Emotion: {features['emotion']}"
                health_text = f"Health: {health_status}"
                distance_text = f"Distance: {distance}m"
                
                cv2.putText(frame, info_text, (x, y-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, emotion_text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, health_text, (x, y+h+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, distance_text, (x, y+h+35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Error processing face: {str(e)}")
                continue
        
        # Add FPS counter
        cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_analysis')
def get_analysis():
    return jsonify(latest_analysis)

if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')
