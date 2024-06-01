import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('lung_cancer_detection_model.h5')



# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input shape of the model
    img = image.resize((256, 256))
    # Convert image to numpy array
    img_array = np.array(img)
    # Normalize the image data
    img_array = img_array / 255.0
    # Expand dimensions to create a batch of size 1
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def make_prediction(image):
    # Preprocess the image
    img_array = preprocess_image(image)
    # Make prediction
    prediction = model.predict(img_array)
    return prediction

# Streamlit app
def main():
    st.title('Lung Cancer Detection')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction when the user clicks the 'Predict' button
        if st.button('Predict'):
            prediction = make_prediction(image)
            st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()
