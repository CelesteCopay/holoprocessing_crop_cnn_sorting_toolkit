import os
import sys
import shutil
import json
import gc
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import config

class UnlabeledDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_paths = []
        
        # Recursively find all images in the raw directory
        for root_dir, dirs, files in os.walk(directory):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    self.image_paths.append(os.path.join(root_dir, f))
                    
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            # If an image is corrupt, return a dummy tensor and the path so we can log it
            return torch.zeros((3, 224, 224)), img_path

def generate_pseudo_labels():
    raw_dir = getattr(config, "UNSORTED_RAW_DIR", "")
    silver_dir = getattr(config, "SILVER_STANDARD_DIR", "")
    threshold = getattr(config, "PSEUDO_LABEL_CONFIDENCE", 0.95)
    model_path = getattr(config, "MODEL_SAVE_PATH", "custom_diatom_resnet.pth")
    classes_path = getattr(config, "CLASSES_SAVE_PATH", "cnn_classes.json")
    
    if not os.path.exists(raw_dir):
        print(f"Error: UNSORTED_RAW_DIR '{raw_dir}' does not exist.")
        return
        
    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        print(f"Error: Model or classes missing. Run train_cnn.py first.")
        return
        
    with open(classes_path, "r") as f:
        classes = json.load(f)
        
    # Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing ResNet-18 on {device}...")
    
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    teacher_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resnet18_teacher.pth")
    model.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"Scanning for images in {raw_dir}...")
    dataset = UnlabeledDataset(raw_dir, transform=transform)
    total_images = len(dataset)
    
    if total_images == 0:
        print("No images found to process!")
        return
        
    print(f"Found {total_images} images. Batching into groups of 64 for VRAM efficiency...")
    
    # Use batch_size 64 to perfectly fit ResNet-152 into a 12GB VRAM GPU
    # Setting num_workers=4 speeds up disk loading significantly
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Create the Silver Standard directory structure
    os.makedirs(silver_dir, exist_ok=True)
    for c in classes:
        os.makedirs(os.path.join(silver_dir, c), exist_ok=True)
        
    print(f"\n--- STARTING PSEUDO-LABELING RUN ---")
    print(f"Target Confidence Filter: >= {threshold*100:.1f}%\n")
    
    processed = 0
    accepted = 0
    reset_every = max(1, int(getattr(config, "MEMORY_RESET_EVERY_N_IMAGES", 500)))
    
    with torch.no_grad():
        for batch_idx, (inputs, paths) in enumerate(dataloader):
            # Check if it's the dummy tensor (sum == 0) meaning corrupt image
            valid_mask = torch.sum(inputs.view(inputs.size(0), -1), dim=1) != 0
            
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probabilities, 1)
            
            # Process each image in the batch
            for i in range(len(paths)):
                if not valid_mask[i]:
                    continue # Skip corrupt images
                    
                prob = max_probs[i].item()
                predicted_class = classes[preds[i].item()]
                img_path = paths[i]
                
                # If we are highly confident, copy it to the Silver Standard
                if prob >= threshold:
                    dest_dir = os.path.join(silver_dir, predicted_class)
                    filename = os.path.basename(img_path)
                    dest_path = os.path.join(dest_dir, filename)
                    
                    # Handle filename collisions just in case
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(dest_dir, f"{base}_{counter}{ext}")
                        counter += 1
                        
                    try:
                        shutil.copy2(img_path, dest_path)
                        accepted += 1
                    except Exception as e:
                        print(f"Error copying {img_path}: {e}")

            del inputs, outputs, probabilities, max_probs, preds, valid_mask
            processed += len(paths)

            if processed % reset_every < len(paths) or processed == total_images:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0 or processed == total_images:
                percent_done = (processed / total_images) * 100
                accept_rate = (accepted / processed) * 100 if processed > 0 else 0
                print(f"[{processed}/{total_images} | {percent_done:.1f}%] - Accepted into Silver Standard: {accepted} ({accept_rate:.1f}% yield)")
                
    print(f"\n--- PSEUDO-LABELING COMPLETE ---")
    print(f"Total processed: {total_images}")
    print(f"Total highly-confident images saved: {accepted}")
    print(f"Your Silver Standard dataset is ready at: {silver_dir}")

if __name__ == "__main__":
    generate_pseudo_labels()
