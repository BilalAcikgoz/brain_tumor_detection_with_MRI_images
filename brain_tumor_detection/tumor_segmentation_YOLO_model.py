from ultralytics import YOLO
from PIL import Image as PILImage

class YOLOTrainer:
    def __init__(self, task='', mode='', model='', imgsz=640, data='',
                 epochs=None, batch=None, learning_rate=None, optimizer=None, weight_decay=None, name='', exist_ok=True):
        self.task = task
        self.mode = mode
        self.model = model
        self.imgsz = imgsz
        self.data = data
        self.epochs = epochs
        self.batch = batch
        self.name = name
        self.exist_ok = exist_ok
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay
    def load_model(self):
        model = YOLO(self.model)
        return model
    def train(self):
        # Train the model
        model = self.load_model()
        model.train(task=self.task, mode=self.mode, data=self.data, epochs=self.epochs, batch=self.batch, imgsz=self.imgsz,
                    name=self.name, exist_ok=self.exist_ok, lr0=self.learning_rate,
                    optimizer=self.optimizer, weight_decay=self.weight_decay)

class YOLOTester:
    def __init__(self, mode='predict', model='', source='', exist_ok=True):
        self.mode = mode
        self.model = model
        self.source = source
        self.exist_ok = exist_ok
    def load_model(self):
        model = YOLO(self.model)
        return model
    def make_prediction(self, img_path):
        model = self.load_model()
        img = PILImage.open(img_path)
        results = model.predict(source=img, save=False)
        return results




























