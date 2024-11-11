import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load face images and labels from your dataset
def load_dataset(data_folder_path):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    for folder_name in os.listdir(data_folder_path):
        folder_path = os.path.join(data_folder_path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            images.append(image)
            labels.append(current_label)

        label_dict[current_label] = folder_name
        current_label += 1

    return np.array(images), np.array(labels), label_dict

# Train Fisherface model
def train_fisherface(images, labels):
    model = cv2.face.FisherFaceRecognizer_create()
    model.train(images, labels)
    return model

# Main code
if __name__ == '__main__':
    # Path to your dataset folder
    dataset_path = 'Train_data'

    # Load the dataset
    images, labels, label_dict = load_dataset(dataset_path)

    # Train test split (optional, for testing accuracy)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Train the Fisherface model
    fisherface_model = train_fisherface(X_train, y_train)

    # Save the trained model
    model_save_path = 'fisherface_model.xml'
    fisherface_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # For testing, you can use the following lines
    predictions = []
    for img in X_test:
        label_predicted, confidence = fisherface_model.predict(img)
        predictions.append(label_predicted)

    # Classification report
    print(classification_report(y_test, predictions))
