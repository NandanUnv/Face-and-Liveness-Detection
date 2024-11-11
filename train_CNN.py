# train of eye blink in cnn

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F  # Importing functional module

# Define the CNN model
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

# Custom Dataset Class
class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.videos = []
        
        for label in ['blink', 'non_blink']:
            folder_path = os.path.join(root_dir, label)
            for video_file in os.listdir(folder_path):
                if video_file.endswith('.mp4'):
                    self.videos.append((os.path.join(folder_path, video_file), label))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path, label = self.videos[idx]
        
        # Read the video and extract frames (you can adjust the frame extraction logic)
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

# Main Training Loop
def train_model():
    dataset = VideoDataset(root_dir='eyeblink_dataset', transform=None)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = EyeBlinkCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'eyeblink_cnn_model.pth')
    print("Model saved as 'eyeblink_cnn_model.pth'")

if __name__ == '__main__':
    train_model()
