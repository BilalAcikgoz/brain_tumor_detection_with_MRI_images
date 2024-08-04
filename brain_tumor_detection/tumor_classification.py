import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from PIL import Image

# Load Dataset and Train-Val-Test Split
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_dataset_from_directory(directory):
    image_paths = []
    labels = []

    class_names = ['Glioma', 'Meningioma', 'Pituitary', 'no_tumor']
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if img_path.endswith(('.png', '.jpg', '.jpeg')):  # Let's filter image file extensions
                image_paths.append(img_path)
                labels.append(class_to_idx[class_name])

    return image_paths, labels


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dir = '/home/bilal-ai/Desktop/anomali_detection_with_medical_images/datasets/brain_dataset/tumor_dataset_for_classification/train'
val_dir = '/home/bilal-ai/Desktop/anomali_detection_with_medical_images/datasets/brain_dataset/tumor_dataset_for_classification/val'
test_dir = '/home/bilal-ai/Desktop/anomali_detection_with_medical_images/datasets/brain_dataset/tumor_dataset_for_classification/test'

train_image_paths, train_labels = load_dataset_from_directory(train_dir)
val_image_paths, val_labels = load_dataset_from_directory(val_dir)
test_image_paths, test_labels = load_dataset_from_directory(test_dir)

train_dataset = CustomDataset(train_image_paths, train_labels, transform=data_transforms['train'])
val_dataset = CustomDataset(val_image_paths, val_labels, transform=data_transforms['val'])
test_dataset = CustomDataset(test_image_paths, test_labels, transform=data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Building ResNet Model and Training
class ResNetWithDropout(nn.Module):
    def __init__(self, original_model, dropout_rate=0.5):
        super(ResNetWithDropout, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        num_ftrs = original_model.fc.in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

original_resnet_model = models.resnet18(pretrained=True)
resnet_model = ResNetWithDropout(original_resnet_model, dropout_rate=0.5)
# num_ftrs = resnet_model.fc.in_features
# resnet_model.fc = nn.Linear(num_ftrs, 4)  # Because there are 4 classes
resnet_model = resnet_model.to(device)

def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_inputs, batch_labels in train_loader:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
    average_loss = total_loss / len(train_loader)
    accuracy = correct / total * 100
    return average_loss, accuracy

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in val_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    average_loss = total_loss / len(val_loader)
    accuracy = correct / total * 100
    return accuracy, average_loss

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet_model.parameters(), lr=0.001, weight_decay=1e-4)

num_epochs = 20
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(resnet_model, train_loader, criterion, optimizer)
    val_accuracy, val_loss = validate(resnet_model, val_loader, criterion)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

print('Eğitim tamamlandı')

# Modeli kaydedin
torch.save(resnet_model, 'resnet_model.pth')

