import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from utils import extract_features

DATASET_PATH = "dataset"

X = []
y = []

for filename in os.listdir(DATASET_PATH):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        label = filename.split("_")[0]  # contoh: 5_abc.png -> label: 5
        img_path = os.path.join(DATASET_PATH, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 100))
        features = extract_features(img)
        X.append(features)
        y.append(label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)



