from ultralytics import YOLO
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLO("yolov8s.pt")
model.to(device)

if __name__ == '__main__':
   results = model.train(
    data="data.yaml", 
    epochs=10,
)

