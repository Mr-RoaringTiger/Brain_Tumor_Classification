from flask import Flask, request, render_template, send_from_directory
import numpy as np
from skimage import io, color, transform
from PIL import Image
from joblib import load
import os

app = Flask(__name__)

# Load the trained Random Forest classifier
rf_classifier = load('random_forest_model.joblib') 

# Function to preprocess the images
def preprocess_image(image):
    # Convert the image to grayscale
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    gray_image = color.rgb2gray(np.array(image))
    # Resize the image to a fixed size (e.g., 128x128)
    resized_image = transform.resize(gray_image, (128, 128))
    # Flatten the image into a 1D array
    flattened_image = resized_image.flatten()
    return flattened_image

# Function to predict on sample image
def predict_sample_image(image, classifier):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Check if the processed image size matches the expected input size
    if len(processed_image) != 16384:
        return "Error: Image size does not match the expected input size."
    
    # Predict on the preprocessed image
    prediction = classifier.predict([processed_image])
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file:
            image = Image.open(file.stream)
            # Predict the category
            prediction = predict_sample_image(image, rf_classifier)
            # Save the uploaded image to display it
            file_path = os.path.join('static', 'uploads', file.filename)
            image.save(file_path)
            return render_template('result.html', prediction=prediction, image_file=file.filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static/uploads', filename)

if __name__ == "__main__":
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    app.run(debug=True)
