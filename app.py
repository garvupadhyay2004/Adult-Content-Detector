import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

MODEL_PATH = "adult_content_detector.h5"

# Load model once
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    non_adult_prob = float(model.predict(img_array)[0][0])
    adult_prob = 1 - non_adult_prob

    if adult_prob >= 0.5:
        return f"🛑 Adult Content Detected\nConfidence: {adult_prob:.4f}"
    else:
        return f"✅ Non-Adult Content\nConfidence: {non_adult_prob:.4f}"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title="Adult Content Detector",
    description="Educational demo for image content classification"
)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True,
    ssr_mode=False
)
