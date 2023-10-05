# Import the necessary libraries
import streamlit as st
import tensorflow as tf
import json
import numpy as np
import io

# Create a function to load the English model
def load_english_model():

    # Load the deep learning model
    english_model = tf.keras.models.load_model('English/Model')

    # Load the tokenizer
    with open('English/sentiment_tokenizer.json') as f:
        english_data = json.load(f)
        english_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(english_data)

    # Return the model and tokenizer
    return english_model, english_tokenizer, 2000

def load_indonesian_model():

    # Load the deep learning model
    indonesian_model = tf.keras.models.load_model('Indonesian/Model')

    # Load the tokenizer
    with open('Indonesian/indonesian_tokenizer.json') as f:
        indonesian_data = json.load(f)
        indonesian_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(indonesian_data)

    # Return the model and tokenizer
    return indonesian_model, indonesian_tokenizer, 500


# Streamlit App
# Define the UI
st.title("Sentiment Classifier")

# Create a dropdown for model selection
model_selection = st.selectbox('Select the Language Model:', ['English', 'Indonesian'])

# Load the model selected by user
if model_selection == 'English':
    model, tokenizer, max_length = load_english_model()
    identifier = 'en'
elif model_selection == 'Indonesian':
    model, tokenizer, max_length = load_indonesian_model()
    identifier = 'id'
else:
    model = None

# Create an input field to receive user's input
user_input = st.text_input('Enter Your Text Here:')

# Define the function to do prediction
def model_prediction(input_text):
    # Tokenize the text
    x = tokenizer.texts_to_sequences([input_text])

    # Pad the tokenized text
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_length, padding='post', truncating='post')

    # Predict the sentiment of the converted text
    result = model.predict(x)

    # Classify and return the result
    if identifier == 'en':
        if result >= 0.5:
            return 'Positive'
        else:
            return 'Negative'

    if identifier == 'id':
        result2 = np.argmax(result)

        if result2 == 0:
            return 'Positive'
        elif result2 == 1:
            return 'Neutral'
        else:
            return 'Negative'

# Create the prediction button and start the inference process
if st.button("Predict"):
    prediction = model_prediction(user_input)
    st.write('Prediction result: ', prediction)
