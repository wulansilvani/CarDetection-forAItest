# CarDetection-forAItest

This repository contains the full implementation of a Car Detection and Classification system developed as part of the ISLAB Technical AI Test 2025. The goal of this project is to detect multiple cars in a video and classify each detected car into a specific category such as MPV, Sedan, SUV, etc.

## Project Overview

This system consists of two main components:
1. **Car Detection Model**: Based on YOLOv5, trained to detect cars in video frames.
2. **Car Classification Model**: A separate deep learning model (ResNet50) trained to classify cropped car images into one of 8 predefined car types.

## Car Types

The classification model supports the following Indonesian car types:
- MPV
- SUV
- Sedan
- Hatchback
- PickUp
- Minibus
- Truck


