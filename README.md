# AI Gender & Age Detection

A web application that uses deep learning to detect gender and age from images. Built with Flask and OpenCV.

## Features

- Real-time gender and age detection
- Modern web interface with drag-and-drop image upload
- Confidence scores for predictions
- Support for multiple faces in a single image
- Responsive design

## Prerequisites

- Python 3.8 or higher
- OpenCV
- Flask
- Required model files (included in the repository)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Gender-and-Age-Detection-from-Image
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload an image through the web interface or use the drag-and-drop feature.

## Model Files

The following model files are required and should be present in the project directory:
- opencv_face_detector.pbtxt
- opencv_face_detector_uint8.pb
- age_deploy.prototxt
- age_net.caffemodel
- gender_deploy.prototxt
- gender_net.caffemodel

## Project Structure

```
├── app.py              # Flask application
├── detect.py           # Core detection logic
├── requirements.txt    # Python dependencies
├── static/            # Static files
│   └── uploads/       # Uploaded and processed images
├── templates/         # HTML templates
│   ├── landing.html   # Landing page
│   └── dashboard.html # Main application page
└── README.md          # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.