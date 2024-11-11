# test of ff model


import cv2
import os
import numpy as np
from sklearn.metrics import classification_report

# Load face images and labels from your dataset (similar to the training function)
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

# Main code for testing
if __name__ == '__main__':
    # Path to your test dataset folder
    test_dataset_path = 'Test_data'

    # Load the test dataset
    test_images, test_labels, label_dict = load_dataset(test_dataset_path)

    # Load the saved Fisherface model
    model_load_path = 'fisherface_model.xml'
    fisherface_model = cv2.face.FisherFaceRecognizer_create()
    fisherface_model.read(model_load_path)
    print(f"Model loaded from {model_load_path}")

    # Test the model with test images
    predictions = []
    for img in test_images:
        label_predicted, confidence = fisherface_model.predict(img)
        predictions.append(label_predicted)

    # Classification report
    print(classification_report(test_labels, predictions, target_names=[label_dict[i] for i in range(len(label_dict))]))
