import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

MODEL_PATH = "adult_content_detector.h5"

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

st.title("Adult Content Detector (Local Demo)")
st.write("Upload an image to classify it as Adult or Non-Adult content.")
st.write("⚠️ This is a local demo for educational purposes.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model predicts NON-ADULT probability
    non_adult_prob = float(model.predict(img_array)[0][0])
    adult_prob = 1 - non_adult_prob

    st.subheader("Prediction Result")

    if adult_prob >= 0.5:
        st.error("🛑 Adult Content Detected")
        st.write("Confidence:", round(adult_prob, 4))
    else:
        st.success("✅ Non-Adult Content")
        st.write("Confidence:", round(non_adult_prob, 4))

