# Brain Tumor Classification

This project focuses on the classification of brain tumors into three categories: Benign, Malignant, and Normal. The dataset used for this classification task was meticulously collected from a local city hospital, ensuring the authenticity and relevance of the data.

## Project Structure

The project includes two main frontend implementations:
1. `rf_classifier_prediction.py`: Utilizes Streamlit to provide an interactive web interface for users to input tumor data and receive predictions.
2. `app.py`: Implements a Flask-based web application, allowing users to interact with the model through a more traditional web server setup.

## Machine Learning Algorithms Utilized

- **Support Vector Machine (SVM) Classifier**: Achieved an accuracy of 87%. This algorithm helps in finding the optimal hyperplane that maximizes the margin between different tumor categories.
- **K-Nearest Neighbors (KNN) Classifier**: Achieved an accuracy of 87%. This instance-based learning algorithm classifies tumors based on the majority class of its nearest neighbors.
- **Decision Tree Classifier**: Achieved an accuracy of 69%. This model uses a tree-like graph of decisions and their possible consequences to classify the tumor types.
- **Random Forest Classifier**: Achieved an accuracy of 87%. This ensemble learning method combines multiple decision trees to improve classification accuracy and robustness.

## Deployment

For deployment, the Random Forest Classifier was selected due to its high accuracy and ability to handle complex data with less risk of overfitting. 

## Libraries Used

The following libraries are used in this project:

```python
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
from flask import Flask, request, jsonify
