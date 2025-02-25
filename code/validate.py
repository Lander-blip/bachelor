from ultralytics import YOLO

# Load a pretrained YOLO model (YOLOv8 in this example)
model = YOLO("/home/fridge/minihack/best.pt")  # You can use 'yolov8s.pt', 'yolov8m.pt', etc.

result=model("/home/fridge/minihack/dataset5/images/train/frame_192.png", save=True)
result[0].show()