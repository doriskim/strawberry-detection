from ultralytics import YOLO

# model = YOLO("yolov8n.pt")
model = YOLO("batch64filtered.pt")
# model = YOLO("colabFiltered.pt")

# train
# model.train(data='../db/strawberryFiltered/data.yaml', batch=32, device='mps')

# real-time detection
# results = model(source=1, show=True, conf=0.5, save=True)

# image detection
model.predict("./filteredjpg", save=True, imgsz=320, conf=0.5)


