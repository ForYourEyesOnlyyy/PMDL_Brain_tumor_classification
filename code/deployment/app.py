import streamlit as st
import requests
from PIL import Image
import io

# Define FastAPI backend URL
FASTAPI_URL = "http://127.0.0.1:8000/predict/"

st.title("Brain Tumor Classification App")
st.write("Upload an MRI image to classify the tumor type.")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI image.', use_column_width=True)

    # Button to send image to FastAPI for prediction
    if st.button("Predict Tumor Type"):
        try:
            # Convert the image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            # Send image to FastAPI
            files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
            response = requests.post(FASTAPI_URL, files=files)

            if response.status_code == 200:
                result = response.json()
                st.write(f"Prediction: {result['prediction']}")
            else:
                st.write(f"Error: {response.json()['error']}")

        except Exception as e:
            st.write(f"Error: {str(e)}")
