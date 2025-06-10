from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from detect import highlightFace

app = Flask(__name__)

# Model parameters
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_MODEL_PATH = 'age_net.caffemodel'
AGE_PROTO_PATH = 'age_deploy.prototxt'
GENDER_MODEL_PATH = 'gender_net.caffemodel'
GENDER_PROTO_PATH = 'gender_deploy.prototxt'

# Age ranges for better interpretation
AGE_RANGES = [
    '(0-2)', '(4-6)', '(8-12)', '(15-20)', 
    '(25-32)', '(38-43)', '(48-53)', '(60-100)'
]

# Ensure upload directory exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

def load_models():
    try:
        age_net = cv2.dnn.readNet(AGE_MODEL_PATH, AGE_PROTO_PATH)
        gender_net = cv2.dnn.readNet(GENDER_MODEL_PATH, GENDER_PROTO_PATH)
        return age_net, gender_net
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None

def get_age_range(age_index, age_confidence):
    """Convert age index to human-readable age range with confidence threshold"""
    if age_confidence < 0.3:  # Low confidence threshold
        return "Unknown"
    
    return AGE_RANGES[age_index]

def process_image(image_path):
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")

        # Load face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with adjusted parameters for better detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return None, "No faces detected in the image"

        # Load age and gender models
        age_net, gender_net = load_models()
        if age_net is None or gender_net is None:
            return None, "Error loading AI models"

        results = []
        for (x, y, w, h) in faces:
            # Expand face region slightly for better detection
            padding = int(min(w, h) * 0.1)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)
            
            face_img = img[y1:y2, x1:x2].copy()
            
            # Skip if face region is too small
            if face_img.size == 0 or min(face_img.shape[:2]) < 20:
                continue
            
            # Prepare blob for gender detection
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Gender detection
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = "Male" if gender_preds[0][0] > gender_preds[0][1] else "Female"
            gender_confidence = float(gender_preds[0][0] if gender == "Male" else gender_preds[0][1])
            
            # Age detection
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age_index = np.argmax(age_preds[0])
            age_confidence = float(age_preds[0][age_index])
            age_range = get_age_range(age_index, age_confidence)
            
            # Calculate overall confidence
            confidence = (gender_confidence + age_confidence) / 2
            
            results.append({
                'gender': gender,
                'age': age_range,
                'confidence': confidence
            })
            
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add text labels with age range
            label = f"{gender}, {age_range}"
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Save processed image
        processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(image_path))
        cv2.imwrite(processed_image_path, img)
        
        return {
            'success': True,
            'results': results,
            'image_url': f'/static/uploads/{os.path.basename(processed_image_path)}'
        }, None

    except Exception as e:
        return None, str(e)

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.content_type.startswith('image/'):
            return jsonify({'error': 'File must be an image'}), 400
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        result, error = process_image(filepath)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True) 