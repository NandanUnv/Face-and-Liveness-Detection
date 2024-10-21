# Face-and-Liveness-Detection
In this project, we worked on Face detection, and liveness refers to eye blinks. The input was live video from a PC camera.
Here we used Fisherface and CNN models for this. Fisherface for face and CNN for eye blink detection.
The output will be provided based on the prediction of both fisherface and CNN models.
If the face and blink are detected output will be face detected and confidence also provided.
If an eye blink is not detected, it will be mentioned in the output terminal. otherwise, no face will be detected.


Packages:

Libraries for Fisherface Face Recognition Model:
* OpenCV
* NumPy
* scikit-learn
* Joblib

Libraries for CNN Eye Blink Detection:
* Torch
* Torchvision
* OpenCV
* NumPy


IDE : Anaconda(Jupyter Notebook)



Datasets:

https://www.kaggle.com/code/faber24/face-anti-spoofing-detection-using-mobilenetv2/input
https://www.blinkingmatters.com/research
