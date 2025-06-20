# === transfer_predict.py ===
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
model = load_model("transfer_cnn_model.h5")
IMG_SIZE = 96

def predict_digit_transfer(image_array):
    """
    Menerima gambar sebagai array RGB dan memprediksi angka menggunakan model transfer learning.
    """
    image = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return np.argmax(prediction)
# === END ===
