import os
import json
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "asl_dataset", "asl_alphabet_train")
MODEL_DIR = os.path.join(BASE_DIR, "models")

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "mobilenet_asl_best.h5")
FINE_TUNED_MODEL_PATH = os.path.join(MODEL_DIR, "mobilenet_asl_finetuned.h5")

# ===============================
# CONFIG
# ===============================
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-5   # VERY IMPORTANT

# ===============================
# DATA GENERATORS
# ===============================
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
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
# LOAD BEST MODEL
# ===============================
print("🔁 Loading best trained model...")
model = load_model(BEST_MODEL_PATH)

# ===============================
# UNFREEZE LAST LAYERS (FINE-TUNING)
# ===============================
base_model = model.layers[0]  # MobileNetV2

for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    layer.trainable = True

print("✅ Fine-tuning last 30 layers of MobileNet")

# ===============================
# RECOMPILE MODEL
# ===============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# CALLBACKS
# ===============================
checkpoint_cb = ModelCheckpoint(
    FINE_TUNED_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop_cb = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ===============================
# FINE-TUNE
# ===============================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, early_stop_cb]
)

print("✅ Fine-tuning completed successfully")
print(f"📦 Saved model: {FINE_TUNED_MODEL_PATH}")
