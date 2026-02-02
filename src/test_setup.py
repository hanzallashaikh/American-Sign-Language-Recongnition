import cv2
import mediapipe as mp
import tensorflow as tf
import tkinter as tk
import os

print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__)
print("TensorFlow:", tf.__version__)
print("Tkinter OK")

print("Dataset folders:", os.listdir("asl_dataset/asl_alphabet_train"))

