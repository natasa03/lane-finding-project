import os
import glob
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAL_DIR = os.path.join(BASE_DIR, 'camera_cal')
OUT_DIR = os.path.join(BASE_DIR, 'output_images', 'corners')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

cols = 9
rows = 6

criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

objectPoints = np.zeros((rows * cols, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

objectPointsArray = []
imgPointsArray = []
failed = []

paths = sorted(glob.glob(os.path.join(CAL_DIR, 'calibration*.jpg')))

for path in paths:
    name = os.path.basename(path)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgSize = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objectPointsArray.append(objectPoints)
        imgPointsArray.append(corners)

        cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
        cv2.imwrite(os.path.join(OUT_DIR, name), img)
        if len(imgPointsArray) == 1:
            cv2.imwrite(os.path.join(DOCS_DIR, 'corners_example.jpg'), img)
    else:
        failed.append(name)

rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, imgSize, None, None)
np.savez(os.path.join(CAL_DIR, 'calib.npz'), mtx=mtx, dist=dist)

error = 0
for i in range(len(objectPointsArray)):
    projected, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
    detected = imgPointsArray[i].reshape(-1, 2).astype(np.float32)
    projected = projected.reshape(-1, 2).astype(np.float32)
    error += cv2.norm(detected, projected, cv2.NORM_L2) / len(projected)

print("Corners found in %d / %d images, skipped: %s" % (len(imgPointsArray), len(paths), failed))
print("RMS reprojection error (px):", rms)
print("Mean error:", error / len(objectPointsArray))
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist.ravel())