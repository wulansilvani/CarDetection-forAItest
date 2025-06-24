import os
import subprocess
import torch

# Cek apakah GPU tersedia dan informasi GPU
if torch.cuda.is_available():
    print("GPU is available:", torch.cuda.get_device_name(0))
else:
    print("GPU not available. Please check your CUDA installation.")

# Set direktori YOLOv5
yolov5_dir = r"C:\Users\wulan\yolov5"
data_yaml = r"C:\Users\wulan\yolov5\bdd100k.yaml"
log_path = r"C:\Users\wulan\CarDetection\train_log_7k.txt"

os.chdir(yolov5_dir)

with open(log_path, "w") as log_file:
    process = subprocess.Popen([
        "python", "train.py",
        "--img", "640",
        "--batch", "16",
        "--epochs", "50",
        "--data", data_yaml,
        "--weights", "yolov5s.pt",
        "--name", "car-detection_e50",
        "--save-period", "1",
        "--device", "0"
    ], stdout=log_file, stderr=subprocess.STDOUT)
    
    process.communicate()

