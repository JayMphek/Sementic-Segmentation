# Safe Segmentation of Drive Scenes in Unstructured Traffic and Adverse Weather

## 📌 Project Overview

This repository contains the end-to-end machine learning pipeline and system integration for the semantic segmentation of driving scenes. The project specifically addresses the visual challenges and performance degradation introduced by adverse weather conditions and unstructured traffic environments. 
---

## ⚠️ Problem Statement
* Autonomous vehicles rely heavily on pixel-level semantic segmentation to recognize critical surroundings like pedestrians, vehicles, and road boundaries.
* State-of-the-art models experience severe performance degradation in adverse conditions such as rain, fog, snow, and low light.
* Visual distortions—including obscured objects, blurred boundaries, glare, and dynamic occlusions—translate directly to high-risk safety hazards.
* Traditional evaluation metrics like mean Intersection over Union (mIoU) fail to adequately penalize dangerous false negatives, such as misclassifying a pedestrian as background.

---

## 📊 Dataset & Evaluation Metric
* **Dataset:** The system utilizes the multi-modal IDD-AW dataset from the ICPR-24 competition, which leverages paired RGB and Near-Infrared (NIR) data channels to maintain visibility through atmospheric scattering.
* **Metric:** Evaluation centers on the novel "Safe mIoU" metric, which explicitly forces safety-conscious development by placing heavily weighted penalties on the misclassification of safety-critical elements.

---

## ⚙️ Proposed Solution & System Architecture

The project implements a complete pipeline split into three core engineering layers:

### 1. Machine Learning Pipelines
Two distinct architectural paradigms are trained and comparatively evaluated:
* **Model Pipeline 1 (U-Net Based):** An encoder-decoder baseline architecture utilizing skip connections to capture high-level context alongside fine spatial details, fused across both RGB and NIR input channels.
* **Model Pipeline 2 (Transformer Based):** A modern Vision Transformer network (such as SegFormer) that uses self-attention mechanisms to capture global context and long-range scene dependencies.
* **Optimization:** Both architectures incorporate a custom loss function engineered to heavily penalize safety-critical errors, directly aligning model behavior with the Safe mIoU objective.

### 2. RESTful Inference Engine
* The champion model is packaged and containerized into a lightweight Flask application.
* The API exposes a single post endpoint (`/predict`) designed to receive an image payload.
* Server-side processing handles raw input formatting (resizing and normalization), feeds the array to the model, and formats the pixel-wise classification matrix into a user-friendly JSON or base64 response.

### 3. Verification Front-End Web Application
* A user-facing demonstration web application is built using Flask’s native templating engine.
* Users submit driving scenes via an intuitive file upload component.
* The web layer fires an asynchronous AJAX request to the backend inference engine to prevent disrupting or reloading the page layout.
* Upon response, the page dynamically overlays a translucent, color-coded class map (e.g., red for pedestrians, blue for vehicles) directly over the original image for immediate human validation.
  
## YouTube Link
https://youtu.be/ORnv1kBYkm4
