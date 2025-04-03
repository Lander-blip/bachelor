from ultralytics import YOLO

model = YOLO('yolo11m.pt')

model.train(data='data.yaml', epochs=20, imgsz=960, batch=0.65)