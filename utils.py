import numpy as np
import cv2

def simulate_colorblindness(img, cb_type):
    """
    Menerapkan matriks transformasi warna untuk mensimulasikan
    penglihatan pengguna dengan buta warna tertentu.
    """
    if cb_type == "protanopia":
        matrix = np.array([[0.567, 0.433, 0],
                           [0.558, 0.442, 0],
                           [0,     0.242, 0.758]])
    elif cb_type == "deuteranopia":
        matrix = np.array([[0.625, 0.375, 0],
                           [0.7,   0.3,   0],
                           [0,     0.3,   0.7]])
    elif cb_type == "tritanopia":
        matrix = np.array([[0.95,  0.05,  0],
                           [0,     0.433, 0.567],
                           [0,     0.475, 0.525]])
    else:
        return img

    img_float = img.astype(np.float32) / 255.0
    transformed = cv2.transform(img_float, matrix)
    result = np.clip(transformed * 255, 0, 255).astype(np.uint8)
    return result
