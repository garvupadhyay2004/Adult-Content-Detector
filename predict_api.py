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
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    confidence = float(model.predict(img_array)[0][0])

    if confidence >= THRESHOLD:
        label = "⚠️ Potential Adult Content"
    else:
        label = "✅ Likely Non-Adult Content"

    return {
        "label": label,
        "confidence": round(confidence, 4)
    }

