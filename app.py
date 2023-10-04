# Import the necessary libraries
import streamlit as st
import tensorflow as tf
import json
import io

# Load the deep learning model
model = tf.keras.models.load_model("Model")

# Load the tokenizer
with open('Tokenizer\sentiment_tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

# Define the function to do prediction
def model_prediction(input_text):
    # Tokenize the text
    x = tokenizer.texts_to_sequences([input_text])

    # Pad the tokenized text
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=2000, padding='post', truncating='post')

    # Predict the sentiment of the converted text
    result = model.predict(x)

    # Classify and return the result
    if result >= 0.5:
        return 'Positive!'
    else:
        return 'Negative!'

# Define the UI
st.title("Sentiment Classifier")

# Create an input field to receive user's input
user_input = st.text_input('Enter Your Text Here!')

# Create the prediction button and start the inference process
if st.button("Predict"):
    prediction = model_prediction(user_input)
    st.write('Prediction result: ', prediction)
