import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "asl_dataset", "asl_alphabet_train")

datagen = ImageDataGenerator(rescale=1.0/255)

generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical"
)

class_indices = generator.class_indices

# Save mapping
with open(os.path.join(BASE_DIR, "models", "class_indices.json"), "w") as f:
    json.dump(class_indices, f, indent=4)

print("✅ class_indices.json saved")
print(class_indices)
