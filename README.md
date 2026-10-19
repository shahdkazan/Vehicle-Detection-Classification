# Vehicle Recognition and Object Detection Using Deep Learning

## Overview

This project develops a deep learning pipeline for vehicle recognition in road scenes, covering image classification, object detection, and model interpretability.

The goal is to identify four vehicle categories — **Bus, Motorcycle, Car, and Truck** — by comparing a custom Convolutional Neural Network (CNN) with transfer learning using MobileNetV2, then training a YOLOv8 model to detect and localize multiple vehicles in an image.

Grad-CAM is also used to visualize the image regions influencing the classification models' predictions.

## Objectives

- Build a custom CNN baseline for four-class vehicle image classification.
- Apply transfer learning and fine-tuning using MobileNetV2 pretrained on ImageNet.
- Train a YOLOv8 object detection model to predict vehicle bounding boxes, classes, and confidence scores.
- Evaluate models using classification metrics and object detection metrics.
- Apply Grad-CAM to improve model interpretability.

## Dataset

The project uses the Vehicles dataset from Kaggle, with separate datasets for classification and object detection.

**Classes:**
- Bus
- Motorcycle
- Car
- Truck

The classification dataset contains **1,773 images**, distributed across the four classes.

Dataset source: [Vehicles Dataset on Kaggle](https://www.kaggle.com/datasets/social888/vehicles/data)

## Methodology

### 1. Data Preparation
- Load and organize image paths and class labels.
- Resize classification images to 224 × 224 pixels.
- Normalize pixel values and prepare TensorFlow data pipelines.
- Create stratified training, validation, and test splits for classification.
- Prepare YOLO-format annotations and separate detection data into training, validation, and test sets.

### 2. Image Classification

**Custom CNN**

A CNN built from scratch using convolutional layers, max pooling, global average pooling, dense layers, and dropout. Adam optimization and early stopping are used during training.

**MobileNetV2 — Transfer Learning**

A pretrained MobileNetV2 backbone is used as a feature extractor with a custom classification head.

The model is trained in two stages:
1. Frozen backbone training.
2. Fine-tuning the later backbone layers using a lower learning rate.

### 3. Object Detection

A pretrained **YOLOv8s** model is fine-tuned on the vehicle detection dataset.

| Configuration | Value |
|---|---|
| Model | YOLOv8s |
| Input image size | 640 × 640 |
| Training epochs | Up to 100 |
| Batch size | 16 |
| Early stopping patience | 10 |

The detector produces bounding boxes, vehicle class labels, and confidence scores. Predictions are also visualized on 20 sampled test images.

### 4. Model Interpretability

Grad-CAM is implemented for the custom CNN and MobileNetV2 classification models to generate heatmaps highlighting image regions that contribute to the predicted class.

These visualizations help investigate model behavior and identify potentially misleading visual features.

## Results

### Classification Performance

| Model | Test Accuracy |
|---|---:|
| Custom CNN | 72.93% |
| MobileNetV2 — Frozen Backbone | 95.49% |
| MobileNetV2 — Fine-Tuned | **96.62%** |

Transfer learning substantially improves classification performance compared with the custom CNN baseline.

### Object Detection Performance

The YOLOv8s model achieved the following results on the detection test split:

| Metric | Score |
|---|---:|
| Precision | 91.56% |
| Recall | 86.63% |
| mAP@50 | **93.10%** |
| mAP@50–95 | 72.88% |

These metrics evaluate the detector's classification and localization performance at different intersection-over-union thresholds.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow, Keras, PyTorch
- **Computer Vision:** OpenCV
- **Object Detection:** Ultralytics YOLOv8
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, scikit-learn
- **Environment:** Kaggle Notebooks with GPU acceleration

## How to Run

1. Open the Jupyter notebook in Kaggle or another compatible notebook environment.
2. Add the [Vehicles dataset](https://www.kaggle.com/datasets/social888/vehicles/data) to the environment.
3. Verify and update the dataset paths in the notebook.
4. Install the required libraries if they are not already available.
5. Run the notebook cells in order to train, evaluate, and visualize the models.

**Note:** The notebook uses Kaggle-specific dataset and output paths. These paths must be updated when running outside Kaggle.

## Key Takeaways

- Compared a custom CNN against pretrained transfer-learning models.
- Improved vehicle classification accuracy through transfer learning and fine-tuning.
- Built and evaluated an object detection pipeline for multiple vehicles in unconstrained scenes.
- Applied Grad-CAM to interpret classification model predictions.
- Developed an end-to-end computer vision workflow spanning data preparation, training, evaluation, and explainability.

## Future Improvements

- Evaluate additional architectures and augmentation strategies.
- Investigate performance under heavy occlusion, poor lighting, and crowded traffic conditions.
- Extend interpretability analysis to object detection predictions.
- Optimize trained models for real-time inference and deployment on edge devices.
