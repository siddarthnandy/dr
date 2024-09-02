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
st.write('Upload images of the fundus for both the right and left eyes. The model will predict the level of DR for each eye.')

# Upload images for the right and left eyes
uploaded_file_right = st.file_uploader("Upload Right Eye Image (Macula Centered)...", type=["jpg", "jpeg", "png"], key='right_eye')
uploaded_file_left = st.file_uploader("Upload Left Eye Image (Macula Centered)...", type=["jpg", "jpeg", "png"], key='left_eye')

# Define class names
class_names = ['0', '1', '2', '3', '4']

def predict_eye(image):
    # Preprocess the image
    image_array = preprocess_image(image)
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return class_names[predicted_class]

# Display and predict for the right eye
if uploaded_file_right is not None:
    st.subheader('Right Eye')
    image_right = Image.open(uploaded_file_right)
    st.image(image_right, caption='Uploaded Right Eye Image', use_column_width=True)
    st.write("Classifying Right Eye...")
    prediction_right = predict_eye(image_right)
    st.write(f"Prediction for Right Eye: {prediction_right}")

# Display and predict for the left eye
if uploaded_file_left is not None:
    st.subheader('Left Eye')
    image_left = Image.open(uploaded_file_left)
    st.image(image_left, caption='Uploaded Left Eye Image', use_column_width=True)
    st.write("Classifying Left Eye...")
    prediction_left = predict_eye(image_left)
    st.write(f"Prediction for Left Eye: {prediction_left}")
