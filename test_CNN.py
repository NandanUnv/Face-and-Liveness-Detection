import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Define the CNN model (same as the one used during training)
class EyeBlinkCNN(nn.Module):
    def __init__(self):
        super(EyeBlinkCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Adjust based on input size
        self.fc2 = nn.Linear(128, 2)  # Binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom Dataset Class for Testing
class TestVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.videos = []
        
        for label in ['blink', 'non_blink']:
            folder_path = os.path.join(root_dir, label)
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Directory {folder_path} does not exist.")
            
            for video_file in os.listdir(folder_path):
                if video_file.endswith('.mp4'):
                    self.videos.append((os.path.join(folder_path, video_file), label))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path, label = self.videos[idx]
        
        # Read the video and extract frames (adjust the frame extraction logic if needed)
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))  # Resize to match model input size
            frames.append(frame)

        cap.release()
        
        # Convert to tensor and normalize
        frames_tensor = np.array(frames).transpose((0, 3, 1, 2))  # Change to (C, H, W)

        if self.transform:
            frames_tensor = self.transform(frames_tensor)

        label_idx = 0 if label == 'blink' else 1
        
        return frames_tensor[0], label_idx  # Return first frame for simplicity

# Function to evaluate the model on the test dataset
def evaluate_model(test_dir):
    model = EyeBlinkCNN()
    model.load_state_dict(torch.load('eyeblink_cnn_model.pth'))  # Load the trained model
    model.eval()  # Set the model to evaluation mode

    dataset = TestVideoDataset(root_dir=test_dir, transform=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    correct_predictions = 0
    total_samples = len(dataset)

    with torch.no_grad():
        for inputs, label in dataloader:
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs.data, 1)
            
            if predicted.item() == label:
                correct_predictions += 1

    accuracy = correct_predictions / total_samples * 100
    print(f'Accuracy on test set: {accuracy:.2f}%')

if __name__ == '__main__':
    test_directory = os.path.join('test_eyeblink')  # Use os.path.join
    evaluate_model(test_directory)
