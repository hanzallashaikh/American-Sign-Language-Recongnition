# American Sign Language Recognition System

## 📌 Overview
This project is a real-time American Sign Language (ASL) recognition system using
deep learning and transfer learning (MobileNetV2).

It captures live video input from a webcam, classifies hand signs, and displays
predictions through a graphical user interface.

## 🚀 Features
- Real-time ASL recognition
- MobileNetV2 with transfer learning
- Fine-tuned CNN model
- GUI built using Tkinter
- High accuracy (~91%)

## 🛠 Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- MobileNetV2
- Tkinter
- NumPy

## 📂 Project Structure
American-Sign-Language-Recognition/
│
├── src/
│   ├── preprocess.py
│   ├── train_mobilenet.py
│   ├── finetune_mobilenet.py
│   ├── evaluate_model.py
│   ├── realtime_predict.py
│   └── gui_app.py
│
├── models/
│   ├── mobilenet_asl_finetuned.h5
│   ├── class_indices.json
│
├── data/
│   ├── processed/        (optional or sample only)
│
├── README.md
├── requirements.txt
└── .gitignore


## ▶️ How to Run
1. Install dependencies
2. Run GUI:
```bash
python src/gui_app.py
