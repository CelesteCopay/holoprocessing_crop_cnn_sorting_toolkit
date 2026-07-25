import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

import config

class MultiDirDataset(Dataset):
    def __init__(self, directories, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = set()
        
        # First pass: find all unique classes
        for d in directories:
            if not os.path.isdir(d):
                continue
            for class_name in os.listdir(d):
                class_dir = os.path.join(d, class_name)
                if os.path.isdir(class_dir):
                    self.classes.add(class_name)
                    
        self.classes = sorted(list(self.classes))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Second pass: collect all images
        for d in directories:
            if not os.path.isdir(d):
                continue
            for class_name in os.listdir(d):
                class_dir = os.path.join(d, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                            self.image_paths.append(os.path.join(class_dir, img_name))
                            self.labels.append(self.class_to_idx[class_name])
                            
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

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

def train_active_learning_cnn(directories, epochs=20, batch_size=128, save_path="custom_diatom_resnet.pth", classes_path="cnn_classes.json"):
    print("INITIALIZING ACTIVE LEARNING RESNET-18...")
    
    # Advanced Data Augmentation to simulate variations of rare particles
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Simulates lighting changes
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
    
    # Assign specific transforms
    train_dataset.dataset.transform = train_transforms
    
    # We must apply val_transforms to val_dataset. 
    # Since random_split creates Subset objects, we wrap it in a custom class to override transforms.
    val_dataset = WrappedSubset(val_dataset, val_transforms)
    
    from torch.utils.data import WeightedRandomSampler, RandomSampler

    # Class-balanced sampling so rare Square Particles appear often enough to learn
    train_labels = [full_dataset.labels[i] for i in train_dataset.indices]
    class_counts = [0] * len(full_dataset.classes)
    for lab in train_labels:
        class_counts[lab] += 1
    class_weights_for_sampler = [
        1.0 / c if c > 0 else 0.0 for c in class_counts
    ]
    sample_weights = [class_weights_for_sampler[lab] for lab in train_labels]

    max_samples = getattr(config, "MAX_SAMPLES_PER_EPOCH", 25000)
    # Keep epochs reasonably long for small datasets without blowing up runtime
    num_epoch_samples = min(max(train_size * 4, 1000), max_samples, 8000)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_epoch_samples,
        replacement=True,
    )
    val_sampler = RandomSampler(
        val_dataset, replacement=True, num_samples=min(val_size, 5000)
    )

    batch_size = 64 if train_size < 1000 else 256
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0, pin_memory=True
    )

    num_classes = len(full_dataset.classes)
    print(f"Classes Map: {full_dataset.class_to_idx}")
    print(f"Class counts (train split): {dict(zip(full_dataset.classes, class_counts))}")
    print(
        f"Training on {num_epoch_samples} images/epoch, "
        f"Validating on {min(val_size, 5000)} images (balanced sampling)."
    )

    with open(classes_path, "w") as f:
        json.dump(full_dataset.classes, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # Inverse-frequency loss weights further emphasize minority class (squares)
    total_train = sum(class_counts) or 1
    loss_weights = torch.tensor(
        [total_train / (num_classes * c) if c > 0 else 0.0 for c in class_counts],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_acc = 0.0
 
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
            
        # Because of Epoch Capping, we only evaluate len(sampler) images per epoch
        epoch_acc = running_corrects.double() / len(sampler)
        
        # Validation
        model.eval()
        val_corrects = 0
        class_correct = {i: 0 for i in range(num_classes)}
        class_total = {i: 0 for i in range(num_classes)}
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                
                # Track per-class accuracy
                for p, l in zip(preds, labels.data):
                    class_total[l.item()] += 1
                    if p.item() == l.item():
                        class_correct[l.item()] += 1
                        
        val_acc = val_corrects.double() / len(val_sampler)
        
        # Calculate Macro-Averaged Accuracy
        class_accuracies = []
        for i in range(num_classes):
            if class_total[i] > 0:
                class_accuracies.append(class_correct[i] / class_total[i])
                
        macro_val_acc = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0.0
        
        scheduler.step(macro_val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f} | Macro Val Acc: {macro_val_acc:.4f}")
        
        if macro_val_acc > best_acc:
            best_acc = macro_val_acc
            torch.save(model.state_dict(), save_path)
            print(f"   -> New Best Macro Acc: {best_acc:.4f}! Model saved.")
            
    print(f"\nTraining Complete! Best Val Acc: {best_acc:.4f}")
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    directories = config.TRAINING_DIRECTORIES
    epochs = config.TRAINING_EPOCHS
    save_path = config.MODEL_SAVE_PATH
    classes_path = config.CLASSES_SAVE_PATH

    print(f"Loading training data from: {directories}")
    train_active_learning_cnn(
        directories, epochs=epochs, save_path=save_path, classes_path=classes_path
    )
