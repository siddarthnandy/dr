import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('color_255_model (4).h5')

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to 28x28
    image = image.resize((28, 28))
    # Convert the image to a numpy array
    image_array = np.array(image).astype('float32')
    # Ensure the pixel values are in the range [0, 255]
    return image_array

# Streamlit app
st.title('Diabetic Retinopathy Grading')
st.write('Upload an image of the fundus and the model will predict the level of DR.')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image_array = preprocess_image(image)
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Define class names
    class_names = ['0', '1', '2', '3', '4']

    # Display the prediction
    st.write(f"Prediction: {class_names[predicted_class]} ")
