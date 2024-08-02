import streamlit as st
import cv2
import random
import numpy as np
from ultralytics import YOLO
import torch
import torchvision

def Detect_Objects_Button(model_path):
    if st.sidebar.button('Detect Tumor'):
        result_image_rgb = None
        if source_img:
            # Running the YOLO model on the uploaded image
            results = model_path(uploaded_image, conf=confidence)
            # colors = [random.choices(range(256), k=3) for _ in classes_ids]
            print(results)
            if results is None:
                st.write("No tumor detected in the image.")
                return None
            else:
                for result in results:
                    if result is not None and result.masks is not None:
                        for mask, box in zip(result.masks.xy, result.boxes):
                            with st.expander("Detection Results"):
                                st.write(box.xywh)
                            points = np.int32([mask])
                            # cv2.polylines(img, points, True, (255, 0, 0), 1)
                            #color_number = classes_ids.index(int(box.cls[0]))
                            result_image_rgb = cv2.fillPoly(uploaded_image, points, color=(255, 0, 0))
                    else:
                        st.write("No tumor detected in the image.")

            with col2:
                if result_image_rgb is not None:
                    st.image(result_image_rgb, caption="Detected Tumor", width=300)

        else:
            st.write("No image uploaded. Please upload an image.")

# Setting page layout
st.set_page_config(
    page_title="Tumor Segmentation Application",  # Setting page title
    page_icon="🤖",  # Setting page icon
    layout="wide",  # Setting layout to wide
    initial_sidebar_state="expanded"  # Expanding sidebar by default
)

brain_yolo_model_path = '/home/bilal-ai/Desktop/tumor_detection_with_medical_images/runs/segment/yolov8m-seg/weights/best.pt'
model_for_brain = YOLO(brain_yolo_model_path)

# Filling difference colors for difference brain tumor types but we have a our brain tumor class
# yolo_classes = list(model.names.values())
# classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
# print(classes_ids)

# Creating sidebar
with (st.sidebar):
    st.header("Image Config")  # Adding header to sidebar
    selections = ['Brain MRI']
    selection = st.selectbox('Please click a selection: ', selections)
    # Adding file uploader to sidebar for selecting images
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("TUMOR SEGMENTATION APP")

# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Reading the uploaded image using OpenCV
        file_bytes = np.asarray(bytearray(source_img.read()), dtype=np.uint8)
        uploaded_image = cv2.imdecode(file_bytes, 1)
        # Converting BGR to RGB for displaying in Streamlit
        uploaded_image_rgb = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)
        # Adding the uploaded image to the page with a caption
        st.image(uploaded_image_rgb, caption="Uploaded Image", width=300)

st.markdown("""
    <style>
        div[data-testid="column"]:nth-of-type(1) {
            padding-right: 20px;
        }
        div[data-testid="column"]:nth-of-type(2) {
            padding-left: 20px;
        }
    </style>
    """, unsafe_allow_html=True)


if selection == 'Brain MRI':
    Detect_Objects_Button(model_for_brain)

