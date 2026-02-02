import os
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "asl_dataset", "asl_alphabet_train")
MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenet_asl_finetuned.h5")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")

# ===============================
# CONFIG
# ===============================
IMAGE_SIZE = 224
BATCH_SIZE = 32

# ===============================
# LOAD MODEL
# ===============================
print("🔁 Loading fine-tuned model...")
model = load_model(MODEL_PATH)

# ===============================
# DATA GENERATOR (ONLY VALIDATION)
# ===============================
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ===============================
# EVALUATE
# ===============================
loss, accuracy = model.evaluate(val_gen)

print("\n✅ MODEL EVALUATION COMPLETE")
print(f"📉 Loss     : {loss:.4f}")
print(f"🎯 Accuracy : {accuracy * 100:.2f}%")
