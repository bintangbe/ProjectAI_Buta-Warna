# === train_transfer.py ===
import os
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from collections import Counter

# --- Konfigurasi ---
DATASET_PATH = "dataset"
IMG_SIZE = 96
EPOCHS = 30
BATCH_SIZE = 16

# --- Load dan preprocess data ---
X = []
y = []

for filename in os.listdir(DATASET_PATH):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        label = filename.split("_")[0]
        if label.isdigit():
            path = os.path.join(DATASET_PATH, filename)
            img = cv2.imread(path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = preprocess_input(img)
            X.append(img)
            y.append(int(label))

# Statistik label
print("Distribusi Label:", Counter(y))

X = np.array(X)
y = to_categorical(np.array(y), num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- Augmentasi ---
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=(0.8, 1.2),
    horizontal_flip=False
)
datagen.fit(X_train)

# --- Model Transfer Learning ---
base_model = MobileNetV2(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0003), loss='categorical_crossentropy', metrics=['accuracy'])

# --- Train initial frozen model ---
model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
          validation_data=(X_test, y_test),
          epochs=10)

# --- Fine-tuning: buka layer akhir MobileNetV2 ---
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])

# --- Lanjutkan training ---
model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
          validation_data=(X_test, y_test),
          epochs=EPOCHS - 10)

# --- Simpan Model ---
model.save("transfer_cnn_model.h5")
print("✅ Model Transfer Learning yang ditingkatkan telah disimpan sebagai transfer_cnn_model.h5")
# === END ===