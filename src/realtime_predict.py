import cv2
import numpy as np
import json
import os
import tensorflow as tf

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenet_asl_finetuned.h5")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")

# ===============================
# CONFIG
# ===============================
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.6

# ===============================
# LOAD MODEL
# ===============================
print("🔁 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# ===============================
# LOAD CLASS LABELS
# ===============================
with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

print("✅ Model and labels loaded")

# ===============================
# PREPROCESS FUNCTION
# ===============================
def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# ===============================
# WEBCAM
# ===============================
cap = cv2.VideoCapture(0)

print("📸 Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = preprocess_frame(frame)
    preds = model.predict(img, verbose=0)
    confidence = np.max(preds)
    class_id = np.argmax(preds)

    if confidence > CONFIDENCE_THRESHOLD:
        label = labels[class_id]
        text = f"{label} ({confidence*100:.1f}%)"
    else:
        text = "No confident prediction"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("ASL Real-Time Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
