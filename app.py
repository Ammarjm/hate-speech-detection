import streamlit as st
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load Model

model = tf.keras.models.load_model(
    "saved_model/hate_speech_rnn.keras"
)


# Load Tokenizer

with open(
    "saved_model/tokenizer.pkl",
    "rb"
) as file:
    tokenizer = pickle.load(file)


# Text Cleaning


def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r'http\S+|www\S+',
        '',
        text
    )

    # Remove punctuation and numbers
    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    # Remove extra spaces
    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()



# Prediction Function

def predict_class(text):

    cleaned = clean_text(text)

    sequence = tokenizer.texts_to_sequences(
        [cleaned]
    )

    padded = pad_sequences(
        sequence,
        maxlen=150,
        padding='post',
        truncating='post'
    )


    prediction = model.predict(
        padded
    )


    predicted_class = prediction.argmax(
        axis=1
    )[0]


    confidence = prediction.max()


    return predicted_class, confidence



# Streamlit Interface

st.title(
    "Hate Speech Detection System"
)


st.write(
    "Enter a text to classify whether it contains hate speech."
)


user_text = st.text_area(
    "Enter text:"
)



if st.button("Predict"):

    if user_text.strip() == "":

        st.warning(
            "Please enter some text."
        )

    else:

        result, confidence = predict_class(
            user_text
        )


        # Mapping classes

        class_names = {
            0: "Hate Speech",
            1: "Normal Speech",
            2: "Offensive Speech"
        }


        st.success(
            f"Predicted Class: {class_names[result]}"
        )


        st.info(
            f"Confidence: {confidence:.2f}"
        )