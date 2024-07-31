#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from skimage import io, color, transform
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from joblib import load
import cv2
import streamlit as st


# In[2]:


# Load the trained Random Forest classifier
rf_classifier = load('random_forest_model.joblib') 


# In[4]:


# Function to preprocess the images
def preprocess_image(image_path):
    # Load the image
    image = io.imread(image_path)
    # Convert the image to grayscale, ignoring alpha channel if present
    if image.shape[-1] == 4:  # Check if the image has an alpha channel
        image = image[:, :, :3]  # Keep only the first three channels (RGB)
    gray_image = color.rgb2gray(image)
    # Resize the image to a fixed size (e.g., 128x128)
    resized_image = transform.resize(gray_image, (128, 128))
    # Flatten the image into a 1D array
    flattened_image = resized_image.flatten()
    return flattened_image

# Function to predict on sample image
def predict_sample_image(sample_image_path, classifier):
    # Preprocess the image
    processed_image = preprocess_image(sample_image_path)
    
    # Check if the processed image size matches the expected input size
    if len(processed_image) != 16384:
        st.error("Error: Image size does not match the expected input size.")
        return None
    
    # Predict on the preprocessed image
    prediction = classifier.predict([processed_image])
    return prediction[0]

# Function to predict on sample image and show it
def predict_and_show_sample(sample_image_path, classifier):
    # Read the sample image
    image = Image.open(sample_image_path)
    
    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict the category
    prediction = predict_sample_image(sample_image_path, classifier)
    
    # Display the prediction
    st.write("Prediction:", prediction)

# Streamlit app
def main():
    # Set title and description
    st.title("Tumor Image Prediction")
    st.write("Upload an image and let the model predict the tumor category.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    # Prediction logic
    if uploaded_file is not None:
        # Save the uploaded image
        image_path = "uploaded_image.png"  # Change this to the appropriate file path
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Predict and show the sample image
        predict_and_show_sample(image_path, rf_classifier)

# Run the app
if __name__ == "__main__":
    main()


# In[ ]:




