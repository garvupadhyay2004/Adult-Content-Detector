import streamlit as st
import tempfile
from predict_api import predict_image

st.set_page_config(
    page_title="Adult Content Detector",
    layout="centered"
)

st.title("🛑 Adult Content Detector")
st.write("Upload an image to check whether it contains adult content.")
st.write("⚠️ This tool is for educational purposes only.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    with st.spinner("Analyzing image..."):
        result = predict_image(temp_path)

    st.subheader("Prediction Result")
    st.write(f"**Label:** {result['label']}")
    st.write(f"**Confidence:** {result['confidence']}")

    if result["confidence"] >= 0.85:
        st.warning("⚠️ This image may contain adult content.")
    else:
        st.success("✅ This image is likely safe.")
