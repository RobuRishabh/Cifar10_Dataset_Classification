# CIFAR-10 Image Classification with Multiple Classifiers and PCA

This project performs image classification on the CIFAR-10 dataset using multiple machine learning classifiers. The dataset consists of 60,000 32x32 color images across 10 classes, and the goal is to predict the class for each image in the test dataset.

## Project Overview

In this project, we use several classifiers to predict the labels of images in the CIFAR-10 dataset. We apply **Principal Component Analysis (PCA)** for dimensionality reduction before training the classifiers. The classifiers used in this project are:

- **Random Forest**
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Classifier (SVC)**

The final prediction for each test image is determined through a **majority voting mechanism** from the predictions of all classifiers.

## Dataset

The CIFAR-10 dataset contains:
- **50,000 training images** and **10,000 test images**.
- The images are labeled into one of **10 classes**: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

### Dataset Classes

The CIFAR-10 dataset includes the following 10 classes:

- `airplane`
- `automobile`
- `bird`
- `cat`
- `deer`
- `dog`
- `frog`
- `horse`
- `ship`
- `truck`

### Dataset Loading

The CIFAR-10 dataset is loaded using TensorFlow's Keras module and TensorFlow Datasets to extract the class names.

## Project Workflow

### 1. **Load CIFAR-10 Dataset**
   - Training and test datasets are loaded using TensorFlow Keras.
   - Each image is a 32x32 color image, flattened to a 1D array (32 * 32 = 1024 pixels).

### 2. **Preprocessing**
   - Convert color images to grayscale by using only the first color channel.
   - Flatten the 2D image arrays into 1D arrays.
   - Apply **PCA (Principal Component Analysis)** to reduce the dimensionality of the dataset, preserving 99% of the variance.

### 3. **Training the Classifiers**
   The following classifiers are trained on the PCA-transformed training data:
   - **RandomForestClassifier**
   - **LogisticRegression**
   - **KNeighborsClassifier**
   - **Support Vector Classifier (SVC)**

### 4. **Prediction on Test Data**
   - Each classifier makes predictions on the PCA-transformed test data.
   - The final prediction for each image is based on a **voting mechanism**, where the most frequent prediction among the classifiers is selected as the final label.

### 5. **Save Predictions**
   - The predictions are saved in a CSV file named `answers.csv`, containing the predicted class names for each test image.

## Prerequisites

The following Python libraries are required:
- `tensorflow`
- `tensorflow_datasets`
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install the required libraries using the following command:
```bash
pip install tensorflow tensorflow-datasets numpy matplotlib scikit-learn
