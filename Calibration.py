import cv2
import numpy as np
import os

# Calibrate
def calibrate():
    # Inner corners: (columns, rows)
    pattern = (4, 4)
    # Real square size, for example 25 mm
    square_size = 25.0
    # Prepare one board's 3D points
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # 3D points in board coordinates
    imgpoints = []  # 2D points in image coordinates

    image_folder = "Calibration images"
    image_size = None

    for filename in os.listdir(image_folder):
        path = os.path.join(image_folder, filename)
        img = cv2.imread(path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCornersSB(gray, pattern)

        if not ret:
            print(f"Not found: {filename}")
            continue

        imgpoints.append(corners.astype(np.float32))
        objpoints.append(objp.copy())

        print(f"Detected: {filename}")

    _, K, _, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    return K