import streamlit as st
from pathlib import Path
from PIL import Image
import tempfile
from src.predict import predict_image

st.set_page_config(page_title="Cervical Cancer Detection and Classification", layout="centered")
st.title("Cervical Cancer Detection and Classification Techniques")
st.write("Upload a Pap smear/cervical cell image. The system performs binary detection and five-class classification.")

image_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
detection_model = "models/detection_model_final.h5"
classification_model = "models/classification_model_final.h5"

if image_file:
    img = Image.open(image_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name)
        tmp_path = tmp.name
    if Path(detection_model).exists() and Path(classification_model).exists():
        result = predict_image(tmp_path, detection_model, classification_model)
        st.subheader("Prediction Result")
        st.write(f"Detection: **{result['detection_result']}** ({result['detection_confidence']:.2%})")
        st.write(f"Classification: **{result['classification_result']}** ({result['classification_confidence']:.2%})")
    else:
        st.warning("Train the models first and place them inside the models folder.")
