#!/usr/bin/env python3
"""
Gender and Age Detection using OpenCV and Deep Learning
This script detects faces in images/video and predicts their gender and age.
"""

import cv2
import math
import argparse
import os
import numpy as np
from typing import Tuple, List, Optional

class FaceDetector:
    def __init__(self, model_dir: str = "."):
        """Initialize the face detector with required models."""
        self.model_dir = model_dir
        self.face_proto = os.path.join(model_dir, "opencv_face_detector.pbtxt")
        self.face_model = os.path.join(model_dir, "opencv_face_detector_uint8.pb")
        self.age_proto = os.path.join(model_dir, "age_deploy.prototxt")
        self.age_model = os.path.join(model_dir, "age_net.caffemodel")
        self.gender_proto = os.path.join(model_dir, "gender_deploy.prototxt")
        self.gender_model = os.path.join(model_dir, "gender_net.caffemodel")

        # Model parameters
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']

        # Load models
        try:
            self.face_net = cv2.dnn.readNet(self.face_model, self.face_proto)
            self.age_net = cv2.dnn.readNet(self.age_model, self.age_proto)
            self.gender_net = cv2.dnn.readNet(self.gender_model, self.gender_proto)
        except Exception as e:
            raise RuntimeError(f"Error loading models: {str(e)}")

    def highlight_face(self, frame: np.ndarray, conf_threshold: float = 0.7) -> Tuple[np.ndarray, List[List[int]]]:
        """Detect faces in the frame and return the frame with highlighted faces."""
        frame_opencv_dnn = frame.copy()
        frame_height = frame_opencv_dnn.shape[0]
        frame_width = frame_opencv_dnn.shape[1]
        
        blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300),
                                   [104, 117, 123], True, False)

        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        face_boxes = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                face_boxes.append([x1, y1, x2, y2])
                
                # Draw rectangle with confidence score
                cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0),
                            int(round(frame_height/150)), 8)
                cv2.putText(frame_opencv_dnn, f"{confidence:.2f}",
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (0, 255, 0), 2, cv2.LINE_AA)
        
        return frame_opencv_dnn, face_boxes

    def predict_age_gender(self, face: np.ndarray) -> Tuple[str, float, str, float]:
        """Predict age and gender for a given face."""
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                   self.MODEL_MEAN_VALUES, swapRB=False)
        
        # Predict gender
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]
        gender_conf = float(gender_preds[0].max())

        # Predict age
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.age_list[age_preds[0].argmax()]
        age_conf = float(age_preds[0].max())

        return gender, gender_conf, age, age_conf

def highlightFace(net, frame, conf_threshold=0.7):
    """
    Detect faces in the frame and return the frame with highlighted faces.
    
    Args:
        net: Face detection network
        frame: Input frame/image
        conf_threshold: Confidence threshold for face detection
        
    Returns:
        tuple: (frame with highlighted faces, list of face bounding boxes)
    """
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3]*frameWidth)
            y1 = int(detections[0,0,i,4]*frameHeight)
            x2 = int(detections[0,0,i,5]*frameWidth)
            y2 = int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
            # Add confidence score
            cv2.putText(frameOpencvDnn, f"{confidence:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
    return frameOpencvDnn, faceBoxes

def main():
    parser = argparse.ArgumentParser(description='Gender and Age Detection')
    parser.add_argument('--image', type=str, help='Path to input image or video file')
    parser.add_argument('--conf-threshold', type=float, default=0.7,
                       help='Confidence threshold for face detection')
    args = parser.parse_args()

    # Model paths
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # Load models
    try:
        faceNet = cv2.dnn.readNet(faceModel, faceProto)
        ageNet = cv2.dnn.readNet(ageModel, ageProto)
        genderNet = cv2.dnn.readNet(genderModel, genderProto)
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return

    # Initialize video capture
    try:
        video = cv2.VideoCapture(args.image if args.image else 0)
        if not video.isOpened():
            raise Exception("Could not open video source")
    except Exception as e:
        print(f"Error initializing video capture: {str(e)}")
        return

    padding = 20
    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            if args.image:  # If processing an image file
                break
            cv2.waitKey()
            continue

        resultImg, faceBoxes = highlightFace(faceNet, frame, args.conf_threshold)
        
        if not faceBoxes:
            print("No face detected")
            cv2.imshow("Detecting age and gender", resultImg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for faceBox in faceBoxes:
            face = frame[max(0,faceBox[1]-padding):
                        min(faceBox[3]+padding,frame.shape[0]-1),
                        max(0,faceBox[0]-padding):
                        min(faceBox[2]+padding, frame.shape[1]-1)]

            if face.size == 0:
                continue

            blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Predict gender
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            genderConf = float(genderPreds[0].max())

            # Predict age
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            ageConf = float(agePreds[0].max())

            # Display results with confidence scores
            label = f"{gender} ({genderConf:.2f}), {age} ({ageConf:.2f})"
            cv2.putText(resultImg, label, (faceBox[0], faceBox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Detecting age and gender", resultImg)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
