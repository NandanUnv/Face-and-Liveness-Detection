# imp code final


import time
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Load Fisherface model
fisherface_model = cv2.face.FisherFaceRecognizer_create()
fisherface_model.read('fisherface_model.xml')

# Define the CNN model (same as during training)
class EyeBlinkCNN(nn.Module):
    def __init__(self):
        super(EyeBlinkCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)  # Binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load CNN model for eye blink detection
cnn_model = EyeBlinkCNN()
cnn_model.load_state_dict(torch.load('eyeblink_cnn_model.pth'))
cnn_model.eval()  # Set to evaluation mode

# Function to detect face using Fisherface model
def detect_face(fisher_model, frame, face_size=(256, 256)):  # Add face size parameter
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        # Resize the face to match the Fisherface model's training size
        face_resized = cv2.resize(face, face_size)
        label, confidence = fisher_model.predict(face_resized)
        
        if confidence < 500:  # Adjust confidence threshold as necessary
            print(f"Face detected with confidence: {confidence}")
            return True, face_resized  # Return True and the resized face ROI if recognized
    return False, None  # Return False if no face detected or recognized

# Function to detect eye blink using CNN model
def detect_blink(cnn_model, face):
    # Convert to RGB and resize for CNN input
    face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    face_resized = cv2.resize(face_rgb, (224, 224))  # Resize to model input size
    face_tensor = np.transpose(face_resized, (2, 0, 1))  # (C, H, W)
    face_tensor = torch.tensor(face_tensor).unsqueeze(0).float()  # Add batch dimension

    with torch.no_grad():
        outputs = cnn_model(face_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item() == 0  # 0 = blink, 1 = no blink

# Main function to combine Fisherface and CNN results
def predict_valid_face(frame):
    # Step 1: Face detection using Fisherface
    face_recognized, face = detect_face(fisherface_model, frame)
    
    if face_recognized:
        # Step 2: Blink detection using CNN
        blink_detected = detect_blink(cnn_model, face)
        
        if blink_detected:
            print("Valid face detected (Face recognized and blink detected).")
            return True
        else:
            print("Face recognized but no blink detected.")
    else:
        print("No valid face detected (Face not recognized).")

    return False  # Return False if it's not a valid face

# Example usage with a video input or webcam
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict valid face
    is_valid = predict_valid_face(frame)
    time.sleep(0.5)
    # Display the result
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
