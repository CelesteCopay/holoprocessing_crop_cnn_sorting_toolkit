import os
import json
import torch
from torchvision import transforms, models
from PIL import Image

import config

def evaluate_gold_standard():
    test_dir = config.EVALUATION_DIR
    model_path = config.MODEL_SAVE_PATH
    classes_path = getattr(config, "CLASSES_SAVE_PATH", None)

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return

    # Prefer training class list so indices match the saved weights
    if classes_path and os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            classes = json.load(f)
    else:
        classes = sorted(
            [
                d
                for d in os.listdir(test_dir)
                if os.path.isdir(os.path.join(test_dir, d))
            ]
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    total = 0
    correct = 0
    class_correct = {c: 0 for c in classes}
    class_total = {c: 0 for c in classes}
    pred_counts = {c: 0 for c in classes}

    print(f"Evaluating on: {test_dir}")
    print(f"Classes: {classes}")
    print(f"Device: {device}")

    with torch.no_grad():
        for class_name in classes:
            class_dir = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: missing class folder {class_dir}")
                continue
            for f in os.listdir(class_dir):
                if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    continue

                img_path = os.path.join(class_dir, f)
                try:
                    image = Image.open(img_path).convert("RGB")
                    tensor = val_transforms(image).unsqueeze(0).to(device)

                    outputs = model(tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    max_prob, preds = torch.max(probabilities, 1)

                    pred_idx = preds.item()
                    pred_class = classes[pred_idx]

                    confidence_threshold = getattr(config, "AMORPHOUS_THRESHOLD", 0.60)
                    if max_prob.item() < confidence_threshold and "Amorphous" in classes:
                        pred_class = "Amorphous"

                    total += 1
                    class_total[class_name] += 1
                    pred_counts[pred_class] += 1

                    if pred_class == class_name:
                        correct += 1
                        class_correct[class_name] += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    print("\n--- RESULTS ---")
    if total == 0:
        print("No images found to evaluate.")
        return

    print(f"Overall Accuracy: {correct}/{total} ({(correct/total)*100:.2f}%)")
    print("\nPer-Class Accuracy (recall):")
    recalls = []
    for c in classes:
        if class_total[c] > 0:
            acc = (class_correct[c] / class_total[c]) * 100
            recalls.append(class_correct[c] / class_total[c])
            print(f"  - {c}: {acc:.2f}% ({class_correct[c]}/{class_total[c]})")
        else:
            print(f"  - {c}: N/A (0 images)")
    if recalls:
        print(f"\nMacro-Averaged Recall: {(sum(recalls)/len(recalls))*100:.2f}%")

    target = "Square Particles"
    if target in classes and pred_counts[target] > 0:
        prec = class_correct[target] / pred_counts[target]
        print(
            f"Square Particles Precision: {prec*100:.2f}% "
            f"({class_correct[target]}/{pred_counts[target]} predicted square)"
        )
    elif target in classes:
        print("Square Particles Precision: N/A (model predicted 0 squares)")

if __name__ == "__main__":
    evaluate_gold_standard()
