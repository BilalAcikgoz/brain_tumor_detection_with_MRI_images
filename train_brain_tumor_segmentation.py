from tumor_segmentation_YOLO_model import YOLOTrainer

# Train
trainer = YOLOTrainer(task='segment', mode='train', model='yolov8m-seg.pt', imgsz=640,
                      data='/home/bilal-ai/Desktop/brain_tumor_detection_with_MRI_images/brain_MRI_dataset/mri_tumor_dataset_for_segmentation/data.yaml',
                      epochs=50, batch=8, learning_rate=0.001, optimizer='Adam', weight_decay=0.001, name='yolov8m-seg', exist_ok=True)
trainer.train()