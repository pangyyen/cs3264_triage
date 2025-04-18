import torch
print(torch.backends.mps.is_available())  # Should return True
print(torch.backends.mps.is_built())      # Should also return True

from ultralytics import YOLO

# Load a pretrained YOLOv8 model (choose variant: n, s, m, l, x)
model = YOLO("yolov8n.pt")  # Try yolov8m.pt or yolov8s.pt for better accuracy

# Train the model
model.train(
    data="/Users/pangyen/Documents/nus mod py/y3s2/cs3264/project/dataset/yolo8_label_roi_1024/mammogram.yaml",  # path to your YAML
    epochs=50,             # adjust depending on your dataset size
    imgsz=1024,             # image size used in preprocessing
    batch=8,                # adjust to fit your GPU memory
    patience=10,            # early stopping
    val=True,  # or False
    project="yolo8_mammo_gpu_50",  # folder name for training logs/weights
    name="exp",             # sub-folder
    exist_ok=True,           # overwrite if folder exists
    device='mps',  # Use Metal Performance Shaders (MPS) for Mac GPUs
    plots=True
)
