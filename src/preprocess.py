import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "asl_dataset", "asl_alphabet_train")

# ===============================
# CONFIG
# ===============================
IMAGE_SIZE = 64
BATCH_SIZE = 32

# ===============================
# IMAGE DATA GENERATORS
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print("✅ Data generators created successfully")
print("Classes:", train_generator.class_indices)
print("Training samples:", train_generator.samples)
print("Validation samples:", val_generator.samples)

# ===============================
# SAVE CLASS LABELS (IMPORTANT)
# ===============================
import numpy as np

CLASS_NAMES = list(train_generator.class_indices.keys())
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
np.save(os.path.join(BASE_DIR, "models", "label_classes.npy"), CLASS_NAMES)

print("✅ Label classes saved")
