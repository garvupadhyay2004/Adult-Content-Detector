import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import ImageFile

# Allow truncated / corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load trained model
model = tf.keras.models.load_model("adult_content_detector.h5")

# Image path
img_path = "test_image.jpg"

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]
confidence = float(prediction)

# Threshold (stricter to reduce false positives)
THRESHOLD = 0.85

# Decision logic
if confidence >= THRESHOLD:
    print("⚠️ Potential Adult Content (Needs Review)")
else:
    print("✅ Likely Non-Adult Content")

# Confidence output
print("Confidence:", round(confidence, 4))

# Confidence interpretation
if confidence >= 0.90:
    print("Confidence Level: HIGH")
elif confidence >= 0.70:
    print("Confidence Level: MEDIUM (manual review recommended)")
else:
    print("Confidence Level: LOW")
