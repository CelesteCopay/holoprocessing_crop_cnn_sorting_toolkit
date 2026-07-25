import gc
import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Import the standalone config instead of .env
import config

class CNNClassifier:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CNNClassifier, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = config.MODEL_SAVE_PATH
        self.classes_path = config.CLASSES_SAVE_PATH
        
        if not os.path.exists(self.model_path) or not os.path.exists(self.classes_path):
            print(f"⚠️ Model or classes file missing. Please run train_cnn.py first.")
            self.model = None
            self.classes = []
            return
            
        with open(self.classes_path, "r") as f:
            self.classes = json.load(f)
            
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.classes))
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self._initialized = True

    @staticmethod
    def release_memory():
        """
        Drop unreferenced Python objects and flush the CUDA caching allocator.
        Call periodically on multi-hundred-thousand image runs to avoid OOM.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def classify(self, image_path: str) -> str:
        """
        Takes an image path and returns the predicted category using the trained CNN.
        """
        if self.model is None:
            return "Background" # Fail-safe fallback
            
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                max_prob, preds = torch.max(probabilities, 1)
                predicted_idx = preds.item()
                confidence = max_prob.item()
                
            predicted_class = self.classes[predicted_idx]
            
            review_threshold = getattr(config, "REVIEW_CONFIDENCE_THRESHOLD", 0.90)
            amorphous_threshold = getattr(config, "AMORPHOUS_THRESHOLD", 0.60)
            
            # If the AI is completely lost, it's Amorphous junk
            if confidence < amorphous_threshold and "Amorphous" in self.classes:
                predicted_class = "Amorphous"
            # If the AI is somewhat confident but not 90%+ sure, send it to human Review
            elif confidence < review_threshold:
                predicted_class = "Review"

            # Explicitly drop per-image tensors so they don't linger until the next GC.
            del input_tensor, outputs, probabilities, max_prob, preds
                
            return predicted_class
            
        except Exception as e:
            print(f"CNN error classifying {image_path}: {e}")
            return "ignore"
