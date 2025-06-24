# CarDetection-forAItest

This repository contains the full implementation of a Car Detection and Classification system developed as part of the ISLAB Technical AI Test 2025. The goal of this project is to detect multiple cars in a video and classify each detected car into a specific category such as MPV, Sedan, SUV, etc.

Output Car Detection: https://drive.google.com/drive/folders/1EV80CM_AOztz6ljPNuC5cG4PV2iHuA0u?usp=drive_link

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

## Note
| Filename                  | Description                                                                                                                                                                                           |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`convertyolov5.py`**    | Script to convert BDD100K JSON annotation format into YOLOv5 `.txt` format for car and motorcycle classes. Used to generate labels for object detection training.                                     |
| **`predict_resnet.py`**   | Performs classification on cropped vehicle images using a trained ResNet50 model. Annotates images with predicted class and confidence, and routes low-confidence predictions to an "Unknown" folder. |
| **`raw_crop.py`**         | Extracts and crops detected vehicles from a CCTV video based on YOLOv5 detection results. Saves each cropped object for classification purposes.                                                      |
| **`scrape_icrawler.py`**  | Downloads vehicle images from Google Images using `icrawler`. Organizes them by class folders (MPV, SUV, Sedan, etc.) to build a classification dataset.                                              |
| **`testing_video.py`**    | Runs object detection (`best.pt`) on a test video (`traffic_test.mp4`), draws bounding boxes, and saves the result as a new annotated video.                                                          |
| **`train.py`**            | Used to launch training of the YOLOv5 detection model using the converted dataset.                                                                                                                    |
| **`train_log_7k.txt`**    | Training log file from the YOLOv5 training process using a subset of 7k images. Stores epoch results, metrics, and notes.                                                                             |
| **`train_resnet.py`**     | Trains a ResNet50 classifier using image data stored in subfolders. Fine-tunes the model to classify vehicle types (MPV, SUV, etc.) and saves the best-performing model.                              |
| **`validate_dataset.py`** | Checks the dataset folder structure and counts number of images per class. Useful to ensure proper dataset formatting before training classification model.                                           |
