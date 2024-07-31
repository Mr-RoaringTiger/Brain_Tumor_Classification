Brain Tumor Classification:
This project focuses on the classification of brain tumors into three categories: Benign, Malignant, and Normal. The dataset used for this classification task was meticulously collected from a local city hospital, ensuring the authenticity and relevance of the data.

Machine Learning Algorithms Utilized:
  Support Vector Machine (SVM) Classifier: Achieved an accuracy of 87%. This algorithm helps in finding the optimal hyperplane that maximizes the margin between different tumor categories.
  K-Nearest Neighbors (KNN) Classifier: Also achieved an accuracy of 87%. This instance-based learning algorithm classifies tumors based on the majority class of its nearest neighbors.
  Decision Tree Classifier: Achieved an accuracy of 69%. This model uses a tree-like graph of decisions and their possible consequences to classify the tumor types.
  Random Forest Classifier: Achieved an accuracy of 87%. This ensemble learning method combines multiple decision trees to improve classification accuracy and robustness.

Deployment:
For deployment, the Random Forest Classifier was selected due to its high accuracy and ability to handle complex data with less risk of overfitting.
The project includes two frontend implementations to interact with the classification model:
rf_classifier_prediction.py: Utilizes Streamlit to provide an interactive web interface for users to input tumor data and receive predictions. Streamlit offers a simple and intuitive way to build web applications for data science projects.
app.py: Implements a Flask-based web application, allowing users to interact with the model through a more traditional web server setup. Flask provides flexibility for building web applications and integrating with various back-end services.
