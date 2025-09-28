import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


# -----------------------------
# 1. Load the trained model
# -----------------------------
MODEL_PATH = r"C:\Users\Divy\OneDrive\Desktop\Projects and Textbooks\Deep-Learning\Projects\Malariya detection\malaria_cnn.h5"  # change to your model path
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# 2. Streamlit UI
# -----------------------------
st.title("Malaria Cell Detection")
st.write("Upload a blood smear image, and the model will predict whether the cell is **Parasitized** or **Uninfected**.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img = img.resize((128, 128))  # <-- replace 128,128 with your model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # only if model was trained on normalized images
    st.write("Image shape for prediction:", img_array.shape)

    # Preprocess image

    # Predict
    predictions = model.predict(img_array)
    
    # For binary classification
    confidence = float(predictions[0][0])
    
    if confidence>0.8 :
        st.success(f"Prediction: **Uninfected** ({(confidence)*100:.2f}% confident)")
        
    else:
        st.success(f"Prediction: **Parasitized** ({1-confidence}% confident)")
