import os
import cv2
import json
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
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

CLASS_NAMES = {v: k for k, v in class_indices.items()}

print("✅ Model and labels loaded")

# ===============================
# PREPROCESS FUNCTION (SAME AS REALTIME)
# ===============================
def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# ===============================
# TKINTER WINDOW
# ===============================
root = tk.Tk()
root.title("ASL Recognition System")
root.geometry("900x700")

title_label = tk.Label(
    root,
    text="American Sign Language Recognition",
    font=("Arial", 20, "bold")
)
title_label.pack(pady=10)

video_label = tk.Label(root)
video_label.pack()

prediction_label = tk.Label(
    root,
    text="Prediction: ---",
    font=("Arial", 18)
)
prediction_label.pack(pady=15)

# ===============================
# WEBCAM
# ===============================
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # Prediction
    img = preprocess_frame(frame)
    preds = model.predict(img, verbose=0)[0]
    confidence = np.max(preds)
    class_id = np.argmax(preds)

    if confidence > CONFIDENCE_THRESHOLD:
        label = CLASS_NAMES[class_id]
        text = f"Prediction: {label} ({confidence*100:.1f}%)"
    else:
        text = "Prediction: No confident sign"

    prediction_label.config(text=text)

    # Show video in GUI
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

# ===============================
# EXIT HANDLER
# ===============================
def on_close():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

# ===============================
# START GUI
# ===============================
update_frame()
root.mainloop()
