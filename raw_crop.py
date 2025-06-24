import torch
import cv2
import os
import sys
sys.path.append(r"C:\Users\wulan")

from pathlib import Path
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox
from yolov5.models.common import DetectMultiBackend
import numpy as np

def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # width, height

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    return coords


video_path = r"C:\Users\wulan\CarDetection\traffic_test.mp4"
weights_path = r"C:\Users\wulan\yolov5\runs\train\car-detection5\weights\best.pt"
output_crop_dir = r"C:\Users\wulan\CarDetection\crops"
img_size = 640
conf_thres = 0.5
iou_thres = 0.45
device = select_device('0')

model = DetectMultiBackend(weights_path, device=device)
model.eval()

os.makedirs(output_crop_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0
crop_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Preprocess frame
    img0 = frame.copy()
    img = letterbox(img0, img_size, stride=32, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(img)).to(device).float() / 255.0
    img = img.unsqueeze(0)

    # Inference
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres, iou_thres)[0]

    # Process detections
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape)
        for i, (*xyxy, conf, cls) in enumerate(pred):
            x1, y1, x2, y2 = map(int, xyxy)
            crop = img0[y1:y2, x1:x2]
            crop_path = os.path.join(output_crop_dir, f"frame{frame_id:05d}_crop{i}.jpg")
            cv2.imwrite(crop_path, crop)
            crop_count += 1

    frame_id += 1

cap.release()
print(f"Selesai. Total crop tersimpan: {crop_count}")
