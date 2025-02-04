# IMDB Movie Review Sentiment Analysis using Simple RNN

## Overview

This project is a deep learning-based Sentiment Analysis tool that classifies IMDB movie reviews as either positive or negative. It uses a Recurrent Neural Network (RNN) model, specifically SimpleRNN, to process text inputs and predict the sentiment of the reviews. The model is trained on the IMDB dataset, a widely used benchmark for sentiment classification.

## Key Features

- Loads and processes the IMDB dataset for movie reviews.
- Uses SimpleRNN with ReLU activation for sentiment analysis.
- Implements text preprocessing including tokenization and padding.
- Includes a Streamlit web application for user-friendly interaction.
- Classifies user-entered reviews into positive or negative sentiment.
- Displays prediction scores indicating confidence in classification.

## Project Structure

- `embedding.ipynb`: Notebook for word embeddings and preprocessing.
- `simplernn.ipynb`: Implementation of the SimpleRNN model.
- `prediction.ipynb`: Notebook for testing predictions on new reviews.
- `main.py`: Streamlit web application for user interaction.
- `simple_rnn_imdb.h5`: Pre-trained RNN model for sentiment analysis.
- `requirements.txt`: List of dependencies for setting up the project.

## Technologies Used

- **Python**: Main programming language.
- **TensorFlow/Keras**: For building and training the deep learning model.
- **Streamlit**: For creating the web application.
- **NumPy**: For numerical operations.
- **IMDB Dataset**: Pre-labeled movie reviews for sentiment analysis.

## Installation & Setup

### Prerequisites

Ensure you have Python installed (preferably Python 3.7 or later). Then, install the required dependencies by running:

```bash
pip install -r requirements.txt
Running the Streamlit Application
To run the Streamlit app, use the following command:

bash
Copy
Edit
streamlit run main.py
How It Works
Load the IMDB word index: Maps words to unique integer values.
Preprocess user input:
Tokenize the text.
Convert words to numerical indices.
Pad the sequence to match the model's input shape.
Load the pre-trained SimpleRNN model.
Predict the sentiment using the model.
Display the results in the Streamlit app, including:
"Positive" or "Negative" classification.
Prediction confidence score.
Expected Output
If the model predicts a score > 0.5, it classifies the review as Positive.
If the model predicts a score ≤ 0.5, it classifies the review as Negative.
Future Enhancements
Train with more advanced architectures like LSTMs or GRUs for better performance.
Extend the model for multi-class sentiment analysis.
Improve the UI with additional visualizations and feedback mechanisms.
