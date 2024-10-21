# Face-and-Liveness-Detection
In this project, we worked on Face detection and liveness refers to eye blink. Input will live vedio from pc camera.
Here we used Fisherface and CNN models for this. Fisherface for face and CNN for eye blink detection.
The output will be provided based on prediction of both fisherface and CNN models.
If the face and blink detected output will be face detected and confidence also provided.
If eye blink is not detected, it will be mentioned in output terminal else will receive no face detected.


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
