from ultralytics import YOLO

model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

model.train(
    data='data/config.yaml',
    epochs=3,
    device='mps'
)

print("DONE!")
