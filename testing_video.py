import sys
sys.path.append(r"C:\Users\wulan")

from pathlib import Path
import cv2
import numpy as np
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.augmentations import letterbox
from yolov5.utils.torch_utils import select_device

video_path = r"C:\Users\wulan\CarDetection\traffic_test.mp4"
weights_path = r"C:\Users\wulan\yolov5\runs\train\car-detection5\weights\best.pt"
output_path = r"C:\Users\wulan\CarDetection\output_detect.avi"
img_size = 640
conf_thres = 0.25
iou_thres = 0.45

device = select_device('0' if torch.cuda.is_available() else 'cpu')

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape."""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords

model = DetectMultiBackend(weights_path, device=device)
model.eval()

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

# PROCESS FRAME BY FRAME
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # PREPROCESS
    img = letterbox(frame, img_size, stride=32)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)  # Add batch dimension
    # INFERENCE
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres, iou_thres)[0]
    # DRAW BOXES
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()

        for *xyxy, conf, cls in pred:
            if conf > 0.5:  # hanya tampilkan yang confidence > 0.5
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # OUTPUT
    out.write(frame)
    cv2.imshow("Car Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
