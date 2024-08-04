from tumor_segmentation_YOLO_model import YOLOTrainer

# Train
trainer = YOLOTrainer(task='segment', mode='train', model='yolov8m-seg.pt', imgsz=640,
                      data='/home/bilal/Desktop/tumor_detection_with_medical_images/datasets/brain_dataset_for_YOLO/data.yaml',
                      epochs=50, batch=8, learning_rate=0.001, optimizer='Adam', weight_decay=0.001, name='yolov8m-seg', exist_ok=True)
trainer.train()