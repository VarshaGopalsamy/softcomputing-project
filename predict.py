from ultralytics import YOLO

model = YOLO(r"C:\Users\varan\Brain-Tumor-Detection-using-YOLOv8\runs\detect\brain_tumor_detector2\weights\best.pt")

model.predict(
    source="test/images",
    conf=0.25,
    save=True
)