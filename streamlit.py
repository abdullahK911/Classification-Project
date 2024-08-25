import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Streamlit UI
st.title("Image Classification with FastAPI")

# Input field for the image URL
image_url = st.text_input("Enter the image URL:")

# Button to make a prediction
if st.button("Predict"):
    if image_url:
        try:
            # Display the image
            response_image = requests.get(image_url)
            image = Image.open(BytesIO(response_image.content))
            st.image(image, caption="Input Image", use_column_width=True)

            # Making a POST request to the FastAPI endpoint
            response_prediction = requests.post(
                "http://127.0.0.1:8000/predicted-image",
                json={"url": image_url}  # Corrected: Use image_url variable here
            )
            
            if response_prediction.status_code == 200:
                # Display the prediction result
                result = response_prediction.json()
                st.write(result)
            else:
                st.write("Error:", response_prediction.status_code, response_prediction.text)
        except Exception as e:
            st.write("An error occurred:", e)
    else:
        st.write("Please enter a valid URL.")
