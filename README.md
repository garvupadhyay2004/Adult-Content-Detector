# Adult Content Detector (Deep Learning Project)

## Overview
This project is an Adult Content Detection system built using deep learning and transfer learning with MobileNetV2. The model classifies images as adult or non-adult content.

## Tech Stack
- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy, SciPy
- PIL

## Project Workflow
1. Dataset validation and preprocessing
2. Image augmentation using ImageDataGenerator
3. Model building using MobileNetV2 (transfer learning)
4. Model training and saving
5. Threshold-based inference for safer predictions

## Key Features
- Transfer learning for fast and efficient training
- Conservative threshold tuning to reduce false positives
- Handles corrupted images gracefully
- Ready for web integration

## How to Run Prediction
```bash
python step5_predict_single_image.py
