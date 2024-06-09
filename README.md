# Email-Spam-Detection
This repository contains the implementation of email spam detection using Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, and a hybrid CNN-LSTM model.<br> <br>

# Introduction
With the increasing volume of emails being exchanged daily, distinguishing between legitimate emails and spam has become a crucial task. Traditional rule-based methods often struggle to keep up with the evolving nature of spam emails. Machine learning-based approaches offer a promising solution by leveraging the power of deep learning to automatically learn and adapt to new spam patterns.<br><br>

# Models Implemented
CNN (Convolutional Neural Network): A CNN model is trained to extract relevant features from the textual content of emails.<br>
LSTM (Long Short-Term Memory): An LSTM network is utilized to capture the sequential nature of language and detect patterns over time.<br>
CNN-LSTM Hybrid Model: This model combines the strengths of both CNN and LSTM architectures, allowing it to capture both local features and long-term dependencies within the email text.<br><br>
# Dataset
The project utilizes a publicly available dataset of labeled emails, with samples of both spam and legitimate emails. The dataset is preprocessed and split into training and testing sets for model training and evaluation.<br><br>

# Usage
Dependencies: Ensure you have the necessary libraries installed, including TensorFlow, Keras, and any other dependencies listed in the requirements.txt file.<br>
Training: Run the training scripts for each model (train_cnn.py, train_lstm.py, train_cnn_lstm.py) to train the models on the provided dataset.<br>
Evaluation: Evaluate the trained models using the evaluation scripts (evaluate_cnn.py, evaluate_lstm.py, evaluate_cnn_lstm.py) on the test set to assess their performance.<br>
Deployment: Once trained, deploy the model of your choice in your preferred environment for real-time spam detection in emails.<br><br>

# Results
The performance metrics of each model, including accuracy, precision, recall, and F1-score, are documented in the repository. Additionally, visualizations such as confusion matrices are provided to further analyze the model's performance.
