import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
from PIL import Image

import config
from train_cnn import MultiDirDataset

def finetune_cnn(directories, target_class, epochs=5, batch_size=32, save_path="custom_diatom_resnet.pth", classes_path="cnn_classes.json"):
    print(f"INITIALIZING FINE-TUNING FOR TARGET CLASS: '{target_class}'...")
    
    if not os.path.exists(save_path) or not os.path.exists(classes_path):
        print(f"Error: Existing model not found at {save_path}. You must run train_cnn.py first before fine-tuning!")
        return

    # Data Augmentation exactly as in training
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = MultiDirDataset(directories, transform=None)
    if len(full_dataset) == 0:
        print("No images found in the specified directories!")
        return
        
    # Split dataset 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.transform = train_transforms
    
    class WrappedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            img_path = self.subset.dataset.image_paths[self.subset.indices[idx]]
            label = self.subset.dataset.labels[self.subset.indices[idx]]
            image = Image.open(img_path).convert("RGB")
            if self.transform: image = self.transform(image)
            return image, label
            
    val_dataset = WrappedSubset(val_dataset, val_transforms)
    
    # Calculate sampler weights with a massive boost to the target class
    train_labels = [train_dataset.dataset.labels[i] for i in train_dataset.indices]
    train_class_counts = Counter(train_labels)
    
    target_class_idx = full_dataset.class_to_idx.get(target_class, -1)
    if target_class_idx == -1:
        print(f"Warning: Target class '{target_class}' not found in the dataset! Proceeding with normal training weights.")
    
    sample_weights = []
    for label in train_labels:
        # Base weight is inverse frequency to naturally balance the classes
        weight = 1.0 / train_class_counts[label]
        
        # If this is the specific category we want to fine-tune on, boost its presence in the batch by 5x!
        if label == target_class_idx:
            weight *= 5.0 
            
        sample_weights.append(weight)
    
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    num_classes = len(full_dataset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # LOAD EXISTING MODEL INSTEAD OF SCRATCH
    print(f"Loading existing brain from {save_path}...")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # MICRO LEARNING RATE: 1e-5 instead of 1e-3 to prevent catastrophic forgetting
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
    
    best_acc = 0.0
    
    print(f"Fine-Tuning on {train_size} images, Validating on {val_size} images for {epochs} epochs.")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_acc = running_corrects.double() / train_size
        
        # Validation
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                
        val_acc = val_corrects.double() / val_size
        
        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            
    print(f"\nFine-Tuning Complete! Best Val Acc: {best_acc:.4f}")
    print(f"Updated model saved to: {save_path}")

if __name__ == "__main__":
    directories = getattr(config, "TRAINING_DIRECTORIES", [])
    target_class = getattr(config, "FINETUNE_TARGET_CLASS", "Amorphous")
    epochs = getattr(config, "FINETUNE_EPOCHS", 5)
    save_path = getattr(config, "MODEL_SAVE_PATH", "custom_diatom_resnet.pth")
    classes_path = getattr(config, "CLASSES_SAVE_PATH", "cnn_classes.json")
    
    if not directories:
        print("No TRAINING_DIRECTORIES found in config.py")
        sys.exit(1)
        
    finetune_cnn(directories, target_class, epochs=epochs, save_path=save_path, classes_path=classes_path)
