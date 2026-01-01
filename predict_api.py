import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import ImageFile

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_PATH = "adult_content_detector.h5"
THRESHOLD = 0.85

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    confidence = float(model.predict(img_array)[0][0])

    if confidence >= THRESHOLD:
        label = "Potential Adult Content"
    else:
        label = "Likely Non-Adult Content"

    return {
        "label": label,
        "confidence": round(confidence, 4)
    }
