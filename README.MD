Road Sign Recognition
This repository contains a Machine Learning project aimed at classifying 20 different types of road signs. The system uses image processing techniques and a Multi-Layer Perceptron (MLP) Neural Network to identify signs based on their visual features.

Project Overview
The goal of this assignment is to build a classifier capable of identifying road signs from a dataset of 1,088 images. A key feature of this project is the comparative analysis of classification performance using Color (RGB) images versus Grayscale images.

Features
Data Preprocessing: Resizing, normalization, and grayscale conversion using OpenCV.
Classification: Implemented using scikit-learn's MLPClassifier.
Comparison: Trains and evaluates two separate models (Color vs. Grayscale) to determine which provides better accuracy.
Visualization: Generates plots comparing predictions and training accuracy.
Requirements
This project uses Python 3. You will need to install the following libraries:

pip install opencv-python numpy scikit-learn matplotlib
How to Run
Clone the repository (or download the ZIP):
bash

git clone https://github.com/Virus-101/Road-Sign-Recognition.git
Navigate to the project folder:
bash

cd Road-Sign-Recognition
Update the Dataset Path:
Open ML.py and update the DATASET_PATH variable to point to the location of the road-signs folder on your computer.
python

DATASET_PATH = r"C:\path\to\your\260202-road-signs\road-signs"
Run the script:
bash

py ML.py
Results
The script trains two models and outputs their respective accuracy on a test set.

Model Type
Input Dimension
Accuracy
MLP (Color)	50x50x3	~85.5% (Example)
MLP (Grayscale)	50x50x1	~82.0% (Example)

Note: Actual results may vary slightly based on the random train/test split.

Dataset
The dataset consists of 20 classes of road signs, organized in the following structure:


road-signs/
    ├── mandatory-chains/
    │   ├── train/
    │   └── test/
    ├── warning-steep-ascent/
    │   ├── train/
    │   └── test/
    ...
Author
Virus-101

License
This project is for educational purposes only.