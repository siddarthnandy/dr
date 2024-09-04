import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# Load the trained model
model = tf.keras.models.load_model('color_255_model (4).h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((28, 28))
    image_array = np.array(image).astype('float32')
    return image_array

# Streamlit app
st.title('Diabetic Retinopathy Grading Report')

# Input fields for report details
patient_name = st.text_input("Patient Name:")
patient_id = st.text_input("Patient ID:")
doctor_name = st.text_input("Doctor Name:")
report_date = st.date_input("Date of Report:")

# Upload images for the right and left eyes
uploaded_file_right = st.file_uploader("Upload Right Eye Image...", type=["jpg", "jpeg", "png"], key='right_eye')
uploaded_file_left = st.file_uploader("Upload Left Eye Image...", type=["jpg", "jpeg", "png"], key='left_eye')

# Define class names and severity levels
class_names = ['0', '1', '2', '3', '4']
severity_levels = {
    '0': 'No DR',
    '1': 'Developing DR',
    '2': 'Moderate DR',
    '3': 'Severe DR',
    '4': 'Very Severe DR'
}

time_to_doctor = {
    '0': 'No immediate need to see a doctor.',
    '1': 'Consult a doctor within 1 month.',
    '2': 'Consult a doctor within 3 weeks.',
    '3': 'Consult a doctor within 1 week.',
    '4': 'Immediate consultation with a doctor is required.'
}

def predict_eye(image):
    image_array = preprocess_image(image)
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return class_names[predicted_class]

# Placeholder for predictions
prediction_right = None
prediction_left = None

# Display images
if uploaded_file_right is not None:
    st.subheader('Right Eye (Macula Centered)')
    image_right = Image.open(uploaded_file_right)
    st.image(image_right, caption='Uploaded Right Eye Image', use_column_width=True)
    prediction_right = predict_eye(image_right)

if uploaded_file_left is not None:
    st.subheader('Left Eye (Macula Centered)')
    image_left = Image.open(uploaded_file_left)
    st.image(image_left, caption='Uploaded Left Eye Image', use_column_width=True)
    prediction_left = predict_eye(image_left)

# Display predictions and generate report at the bottom
if prediction_right is not None or prediction_left is not None:
    st.subheader('Predictions')
    if prediction_right is not None:
        st.write(f"Prediction for Right Eye: {prediction_right} - {severity_levels[prediction_right]}")
    if prediction_left is not None:
        st.write(f"Prediction for Left Eye: {prediction_left} - {severity_levels[prediction_left]}")

    # Generate the report as a PDF
    def generate_pdf_report(image_right, image_left, prediction_right, prediction_left, patient_name, patient_id, doctor_name, report_date):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(200, height - 50, "Diabetic Retinopathy Diagnosis")

        # Report details
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Name: {patient_name}")
        c.drawString(300, height - 80, f"Date: {report_date}")
        c.drawString(50, height - 100, f"Participant ID: {patient_id}")
        c.drawString(300, height - 100, f"Doctor: {doctor_name}")

        # Images and labels
        c.drawString(100, height - 150, "Right Eye (Macula Centered)")
        c.drawString(350, height - 150, "Left Eye (Macula Centered)")
        c.drawImage(ImageReader(image_right), 50, height - 400, width=3*inch, height=3*inch)
        c.drawImage(ImageReader(image_left), 300, height - 400, width=3*inch, height=3*inch)

        # Results table
        c.drawString(50, height - 450, "Results")
        c.line(50, height - 460, width - 50, height - 460)
        c.drawString(50, height - 480, "Diabetic Retinopathy")
        c.drawString(250, height - 480, f"{severity_levels[prediction_right]}")
        c.drawString(450, height - 480, f"{severity_levels[prediction_left]}")

        

        # Referral Advice
        c.drawString(50, height - 540, "Referral Advice (Please refer to the following time for ophthalmology review)")
        c.line(50, height - 550, width - 50, height - 550)

        c.drawString(50, height - 570, "Immediate")
        c.drawString(150, height - 570, "1 week")
        c.drawString(250, height - 570, "3 weeks")
        c.drawString(350, height - 570, "5 weeks")
        c.drawString(450, height - 570, "1 year (For general eye checkup)")

        # Mark the appropriate referral time based on the predictions
        if max(prediction_right, prediction_left) == '4':
            c.rect(50, height - 590, 50, 20, fill=1)
        elif max(prediction_right, prediction_left) == '3':
            c.rect(150, height - 590, 50, 20, fill=1)
        elif max(prediction_right, prediction_left) == '2':
            c.rect(250, height - 590, 50, 20, fill=1)
        elif max(prediction_right, prediction_left) == '1':
            c.rect(350, height - 590, 50, 20, fill=1)
        else:
            c.rect(450, height - 590, 50, 20, fill=1)

        # Disclaimers and Clinician notes
        c.drawString(50, height - 630, "Disclaimers:")
        c.drawString(50, height - 650, "This report does not replace professional medical advice, diagnosis or treatment.")
        c.drawString(50, height - 670, "Clinician notes:")

        c.save()
        buffer.seek(0)
        return buffer

    # Provide a download button
    if st.button("Download DR Report"):
        pdf_report = generate_pdf_report(image_right, image_left, prediction_right, prediction_left, patient_name, patient_id, doctor_name, report_date)
        st.download_button(label="Download Report", data=pdf_report, file_name="DR_Grading_Report.pdf", mime="application/pdf")
