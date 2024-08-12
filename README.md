BRAIN TUMOR SEGMENTATION AND CLASSIFICATION

Hello everyone! The aim of this project is to segment the areas where the tumor is located on brain MRI images and also to find out which class this tumor belongs to. We have 4 classes in total: Glioma, Meningioma, Pituitary, No Tumor

ABOUT THE DATA AND METHOD

We can divide our dataset into two as classification and segmentation. The classification dataset contains train-val-test folders and each of them contains the 4 classes mentioned above. We perform our classification process using this dataset. We use the resnet18 architecture for classification and the success rate is 98% on both the training and validation datasets.

We took the segmentation dataset from the classification dataset and labeled this data for segmentation via roboflow. Thanks to the automatically generated 'yaml' file, we easily gave our dataset to our YOLOv8-seg model.

ABOUT THE APPLICATION

We wrote the interface of our application with the streamlit library and transferred our models to our application with this library and were able to make predictions about our model. Since we have two separate models, when the button is pressed, both models will work and apply both segmentation and classification processes on the image. A short demonstration of the application is given below.

![Screenshot from 2024-08-12 12-33-36](https://github.com/user-attachments/assets/b8132dad-5fe4-4f45-b529-abbd6e6272e1)
