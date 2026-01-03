import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import gradio as gr

MODEL_PATH = "adult_content_detector.h5"

# Load model once
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict_image(img):
    if img is None:
        return "No image uploaded"

    # Ensure RGB
    img = img.convert("RGB")

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model predicts NON-ADULT probability
    non_adult_prob = float(model.predict(img_array)[0][0])
    adult_prob = 1 - non_adult_prob

    if adult_prob >= 0.5:
        return f"🛑 Adult Content Detected\nConfidence: {adult_prob:.4f}"
    else:
        return f"✅ Non-Adult Content\nConfidence: {non_adult_prob:.4f}"

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Prediction Result"),
    title="Adult Content Detector",
    description="Upload an image to classify it as Adult or Non-Adult content.\n⚠️ Educational purposes only."
)

demo.launch()
