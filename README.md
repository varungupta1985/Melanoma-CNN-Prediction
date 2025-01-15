## Assignment : Melanoma Skin Cancer Detection using CNN
## Optimizing Multiclass Image Classification with Custom CNNs in TensorFlow
1. Skin Cancer detection from images  using CNN, Tensorflow 
2. This repository contains the code for detecting melanoma using a Convolutional Neural Network (CNN) and TensorFlow with GPU acceleration. The model is trained on a dataset of skin lesion images to classify them as either melanoma or benign.

## Table of Contents
* [General Info](#general-information)
* [DataSet](#DataSet)
* [Business Goal](#business-goal)
* [Business Risk](#business-risk)
* [Project Pipeline](#proj_pipeline)
* [Model](#model)
* [Usage](#usage)
* [Conclusions](#conclusions)
* [Technologies Used](#technologies-used)
* [References](#References)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Information
- Melanoma is a type of skin cancer that can be deadly if not detected early. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce manual effort in diagnosis.
- The project is aimed at developing a CNN-based model for detecting melanoma in skin images. The dataset used in the project consists of 2357 images of malignant and benign oncological diseases, collected from the International Skin Imaging Collaboration (ISIC).
- The aim is to develop a model that can accurately detect melanoma in skin images and reduce manual efforts in diagnosis.
- The business problem that the project is trying to solve is the manual effort required in the diagnosis of melanoma, a type of skin cancer. The manual process of evaluating images to detect melanoma can be time-consuming and prone to human error. The aim of the project is to develop a CNN-based model that can accurately detect melanoma in skin images and reduce the manual effort required in the diagnosis process. The deployment of the model in a dermatologist's workflow has the potential to increase efficiency and accuracy, potentially leading to better patient outcomes.
 
## DataSet : 
The model is trained on the ISIC Archive dataset, which contains a large number of dermoscopic images of skin lesions, including both benign and malignant melanomas. The dataset is pre-processed and split into training and validation sets.

        * Actinic keratosis
        * Basal cell carcinoma
        * Dermatofibroma
        * Melanoma
        * Nevus
        * Pigmented benign keratosis
        * Seborrheic keratosis
        * Squamous cell carcinoma
        * Vascular lesion


## Business Goal:
The objective is to construct a multiclass classification model using a personalized convolutional neural network (CNN) implemented in TensorFlow.


## Business Risk:
Some of the business risks associated with the project are:
1. The model's accuracy in detecting melanoma in skin images is a crucial factor. If the model produces incorrect results, it could lead to misdiagnosis and harm to patients.

2. The quality and reliability of the data used to train the model can have a significant impact on its performance. Any errors or biases in the data set can result in inaccurate results.

3. Developing a CNN-based model can be technically challenging, and there may be difficulties in training the model and optimizing its performance.

4. There may be other existing solutions or competing models in the market that perform similarly or better, making it difficult to gain a competitive advantage.

## Model
The model uses a simple CNN architecture with convolutional, Max pooling and dense layers. The model is trained using SparseCategoricalCrossentropy loss and the Adam optimizer.

 
## Conclusion 
The model showed promising results in detecting melanoma in skin images with high accuracy, sensitivity, and specificity. The deployment of the model in a dermatologist's workflow has the potential to reduce manual effort in the diagnosis of melanoma.

## Prepared By
[Varun Gupta] (https://github.com/varungupta1985/)


