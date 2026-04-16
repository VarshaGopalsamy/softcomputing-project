import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("Brain Tumor Localization using YOLOv8")

model = YOLO(r"C:\Users\varan\Brain-Tumor-Detection-using-YOLOv8\runs\detect\brain_tumor_detector2\weights\best.pt")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_file.name)

    if st.button("Detect Tumor"):

        results = model.predict(source=temp_file.name, conf=0.25)

        output_image = results[0].plot()

        if len(results[0].boxes) > 0:
            st.success("Tumor Detected with Bounding Box")
        else:
            st.warning("No Tumor Detected")

        st.image(output_image, caption="Tumor Localization Result")