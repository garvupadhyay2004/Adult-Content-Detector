import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import BatchNormalization
from PIL import ImageFile

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------
# Fix BatchNormalization axis issue
# -------------------------
class FixedBatchNormalization(BatchNormalization):
    def __init__(self, axis=3, **kwargs):
        if isinstance(axis, (list, tuple)):
            axis = axis[0]
        super().__init__(axis=axis, **kwargs)

# -------------------------
# Config
# -------------------------
MODEL_PATH = "adult_content_detector.keras"
THRESHOLD = 0.85

# -------------------------
# Load model ONCE
# -------------------------
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "BatchNormalization": FixedBatchNormalization
    },
    compile=False
)

# -------------------------
# Prediction function
# -------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    non_adult_prob = float(model.predict(img_array)[0][0])
    adult_prob = 1 - non_adult_prob

    if adult_prob >= 0.5:
        label = "Adult Content"
        confidence = adult_prob
    else:
        label = "Non-Adult Content"
        confidence = non_adult_prob

    return {
        "label": label,
        "confidence": round(confidence, 4)
    }


