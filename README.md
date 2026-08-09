# Hate Speech Detection Using Deep Learning

This project develops a hate speech detection system using Natural Language Processing and Deep Learning.

## Models

- Simple RNN
- LSTM
- Bidirectional RNN

## Final Model

The Bidirectional RNN achieved the best overall performance.

Accuracy: 85.64%

F1-Score: 86.62%

## Dataset

The project uses a labeled hate speech text dataset.

## Preprocessing

- Lowercase conversion
- URL removal
- Punctuation removal
- Number removal
- Stop-word removal
- Tokenization
- Lemmatization

## Class Imbalance

Class weights were used to handle class imbalance.

## Deployment

The application was developed using Streamlit.

Run the application with:

streamlit run app.py