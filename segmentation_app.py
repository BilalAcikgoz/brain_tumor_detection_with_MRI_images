import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ResNetWithDropout(nn.Module):
    def __init__(self, original_model, dropout_rate=0.5):
        super(ResNetWithDropout, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        num_ftrs = original_model.fc.in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, 4)
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

brain_classification_model_path = '/home/bilal-ai/Desktop/brain_tumor_detection_with_MRI_images/resnet_model.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
original_resnet_model = models.resnet18(pretrained=True)
classification_model_for_brain = ResNetWithDropout(original_resnet_model, dropout_rate=0.5)
classification_model_for_brain = torch.load(brain_classification_model_path, map_location=device)
classification_model_for_brain.to(device)
classification_model_for_brain.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
def classify_image(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = classification_model_for_brain(image)
    _, predicted = torch.max(outputs, 1)
    class_names = ['Glioma', 'Meningioma', 'Pituitary', 'no_tumor']
    return class_names[predicted.item()]
def Detect_Objects_Button(detection_model_path):
    if st.sidebar.button('Detect Tumor'):
        result_image_rgb = None
        if source_img:
            # Running the YOLO model on the uploaded image
            results_detection = detection_model_path(uploaded_image, conf=confidence)
            print(results_detection)
            if results_detection is None:
                st.write("No tumor detected in the image.")
                return None
            else:
                for result in results_detection:
                    if result is not None and result.masks is not None:
                        for mask, box in zip(result.masks.xy, result.boxes):
                            points = np.int32([mask])
                            # cv2.polylines(img, points, True, (255, 0, 0), 1)
                            # color_number = classes_ids.index(int(box.cls[0]))
                            result_image_rgb = cv2.fillPoly(uploaded_image, points, color=(255, 0, 0))
                    else:
                        st.write("No tumor detected in the image.")

            with col2:
                if result_image_rgb is not None:
                    st.image(result_image_rgb, caption="Detected Tumor", width=300)

            # Classification
            pil_image = Image.fromarray(cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB))
            classification_result = classify_image(pil_image)
            st.write(f'Classification Result: {classification_result}')

        else:
            st.write("No image uploaded. Please upload an image.")

# Setting page layout
st.set_page_config(
    page_title="Brain Tumor Segmentation and Classification Application",  # Setting page title
    page_icon="🤖",  # Setting page icon
    layout="wide",  # Setting layout to wide
    initial_sidebar_state="expanded"  # Expanding sidebar by default
)

brain_yolo_model_path = '/home/bilal-ai/Desktop/brain_tumor_detection_with_MRI_images/runs/segment/yolov8m-seg/weights/best.pt'
detection_model_for_brain = YOLO(brain_yolo_model_path)

# Creating sidebar
with (st.sidebar):
    st.header("Image Config")  # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("Brain Tumor Segmentation and Classification Application")

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

Detect_Objects_Button(detection_model_for_brain)

