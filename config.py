# STANDALONE CNN CONFIGURATION
# Just edit these paths to point to the folders on your computer!
# Make sure to use 'r' before the quotes so Windows paths work correctly (e.g., r"C:\path")

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


TRAINING_DIRECTORIES = [
    os.path.join(BASE_DIR, "coiled_elongated_diatom_train_data"),
]

# Holdout / NewGoldStandard-based binary eval set
EVALUATION_DIR = os.path.join(BASE_DIR, "coiled_elongated_diatom_eval_data")

# Where to save and load the trained AI brain
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "coiled_elongated_diatom_resnet18.pth")

# Where to save and load the list of categories the AI knows
CLASSES_SAVE_PATH = os.path.join(BASE_DIR, "coiled_elongated_diatom_cnn_classes.json")

# ADVANCED TUNING
# Number of times the AI studies the entire dataset during training
TRAINING_EPOCHS = 25

# Any image the AI predicts with confidence lower than this will be sent to the 'Review' folder
REVIEW_CONFIDENCE_THRESHOLD = 0.90

# If the AI's confidence is severely low, it is likely junk and sorted to Amorphous
AMORPHOUS_THRESHOLD = 0.45

# --- FINE TUNING ---
# The specific category you want the AI to focus on learning better
FINETUNE_TARGET_CLASS = "Square Particles"

# --- PSEUDO LABELING (TEACHER-STUDENT) ---
# Folder containing massive amounts of unsorted raw images
UNSORTED_RAW_DIR = r"F:\Users\myfri\Downloads\transfer"

# Where the highly-confident AI predictions will be saved
SILVER_STANDARD_DIR = r"F:\Users\myfri\Downloads\SilverStandard"

# Only accept predictions if the AI is this confident (0.0 to 1.0)
PSEUDO_LABEL_CONFIDENCE = 0.95

# --- MASSIVE DATASET TUNING ---
# Caps the total number of images the AI looks at per epoch.
MAX_SAMPLES_PER_EPOCH = 80000

# How many times the AI studies the dataset during fine-tuning (keep this low to prevent regression)
FINETUNE_EPOCHS = 5

# --- LONG-RUN MEMORY SAFETY (auto_sorter / mega batches) ---
# Force Python GC + CUDA cache flush every N images to avoid OOM on ~1M runs.
# 500 ≈ 2000 resets on a million-image job; lower if you still OOM, raise if throughput dips.
MEMORY_RESET_EVERY_N_IMAGES = 2000
