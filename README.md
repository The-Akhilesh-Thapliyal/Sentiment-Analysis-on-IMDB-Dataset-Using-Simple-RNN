# Sentiment Analysis on IMDB Dataset Using Simple RNN

![Project Banner](https://github.com/user-attachments/assets/f0495765-35c9-4fdf-8811-35d6a6a901fd)


This project implements a sentiment analysis model using a Simple Recurrent Neural Network (RNN) on the IMDB movie review dataset. The goal is to predict whether a given movie review is positive or negative. The project also includes a user-friendly Streamlit app for real-time sentiment analysis.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Streamlit Application](#streamlit-application)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction
This project aims to develop a sentiment analysis model for movie reviews using a Simple RNN. The model is trained on the IMDB dataset, which contains 50,000 movie reviews labeled as positive or negative. The project also includes a Streamlit application that allows users to input movie reviews and receive real-time sentiment predictions.

## Dataset
The IMDB dataset is used in this project, containing 25,000 training reviews and 25,000 test reviews. The reviews are preprocessed and tokenized, with a maximum vocabulary size of 10,000 words.

## Model Architecture
The model architecture includes:
- An Embedding layer to convert words into dense vectors.
- Two Simple RNN layers, with the first RNN layer returning sequences and a Dropout layer for regularization.
- A Dense layer with a sigmoid activation function for binary classification.

### Model Summary
This screenshot provides a summary of the model architecture, including the layers and their configurations.
<br><br>
![Model_Summary](https://github.com/user-attachments/assets/3194a947-e67d-41cd-94f0-08f0fc272f69)

## Training the Model
The model was trained using the binary cross-entropy loss function and the Adam optimizer. Early stopping was implemented to prevent overfitting.

### Training and Validation Accuracy
The plot below illustrates the training and validation accuracy over the epochs, showing how the model improved during training and how it performed on unseen validation data.
<br><br>
![Training_Validation_Accuracy](https://github.com/user-attachments/assets/7a0e2527-de18-4759-bfed-ba3aaaa0cd28)

## Model Evaluation
The model's performance was evaluated on the test set. The following plot shows the distribution of review lengths in the dataset.

### Review Length Distribution
The histogram below shows the distribution of review lengths in the IMDB dataset, helping us understand the variability in review lengths before proceeding to model training.
<br><br>
![Review_Length_Distribution](https://github.com/user-attachments/assets/630ed770-eb5b-426d-b4a1-3fef52168379)

## Streamlit Application
The project includes a Streamlit application where users can input a movie review and receive a sentiment analysis (positive or negative) along with the prediction score.

### Streamlit App

**Positive Review :** 
This screenshot captures the output when a positive review is analyzed by the Streamlit app. The app correctly identifies the sentiment as positive and displays the prediction score.
<br><br>
![Positive_Review](https://github.com/user-attachments/assets/b7991a25-0803-43ce-9bab-a0e9995de377)

**Negative Review :**
Similarly, this screenshot shows the app's output when a negative review is analyzed, demonstrating the model's ability to correctly predict a negative sentiment.
<br><br>
![Negative_Review](https://github.com/user-attachments/assets/ef34b81d-9b08-462b-9825-e5f48c7f4f67)

## Installation

To run the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/imdb-sentiment-analysis.git
    cd imdb-sentiment-analysis
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run main.py
    ```

## Usage
- Enter a movie review into the text area provided in the Streamlit app.
- Click "Analyze" to see the sentiment prediction and the prediction score.

## Screenshots
- **Review Length Distribution:** <br><br> ![Review_Length_Distribution](https://github.com/user-attachments/assets/630ed770-eb5b-426d-b4a1-3fef52168379)
- **Model Summary:** <br><br> ![Model_Summary](https://github.com/user-attachments/assets/3194a947-e67d-41cd-94f0-08f0fc272f69)
- **Training and Validation Accuracy:** <br><br> ![Training_Validation_Accuracy](https://github.com/user-attachments/assets/7a0e2527-de18-4759-bfed-ba3aaaa0cd28)
- **Streamlit App:** <br><br> ![Positive_Review](https://github.com/user-attachments/assets/b7991a25-0803-43ce-9bab-a0e9995de377)

## Results
The Simple RNN model achieved high accuracy on both training and validation datasets. The Streamlit app provides an intuitive interface for users to analyze movie reviews and obtain real-time sentiment predictions.

## Conclusion
This project showcases the implementation of a sentiment analysis model using a Simple RNN. It highlights the importance of data preprocessing, model architecture tuning, and regularization techniques in developing effective deep learning models.
