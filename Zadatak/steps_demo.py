import os
import glob
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
import lane_pipeline as lp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'output_images')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')
EXAMPLE = 'test2'

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'histogram'), exist_ok=True)


def to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def pair(left, right, left_text, right_text):
    left, right = to_bgr(left), to_bgr(right)
    cv2.putText(left, left_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
    cv2.putText(right, right_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
    return np.hstack((left, right))


def save(step, name, image):
    folder = os.path.join(OUT_DIR, step)
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, name + '.jpg'), image)


def to_docs(step, name, docs_name):
    shutil.copy(os.path.join(OUT_DIR, step, name + '.jpg'), os.path.join(DOCS_DIR, docs_name))


for path in sorted(glob.glob(os.path.join(BASE_DIR, 'camera_cal', '*.jpg'))):
    name = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path)
    save('undistort', name, pair(img, lp.undistort(img), 'Original', 'Undistorted'))

for path in sorted(glob.glob(os.path.join(BASE_DIR, 'test_images', '*.jpg'))):
    name = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path)
    h, w = img.shape[:2]

    undist = lp.undistort(img)
    save('undistort', name, pair(img, undist, 'Original', 'Undistorted'))

    binary = lp.threshold_binary(undist)
    save('threshold', name, pair(undist, binary, 'Undistorted', 'Binary'))

    src, dst = lp.get_points(w, h)
    annotated = undist.copy()
    cv2.polylines(annotated, [src.astype(np.int32)], True, (0, 255, 0), 3)
    warped_color = lp.warper(undist, src, dst, cv2.INTER_LINEAR)
    for x in (int(dst[0][0]), int(dst[2][0])):
        cv2.line(warped_color, (x, 0), (x, h), (0, 0, 255), 2)
    save('perspective', name, pair(annotated, warped_color, 'Trapezoid', "Bird's eye view"))

    warped = lp.warper(binary, src, dst)
    left_fit, right_fit, info = lp.find_lane_lines(warped)
    save('lanes', name, pair(warped, lp.draw_search(warped, left_fit, right_fit, info), "Bird's eye binary", 'Lane lines'))

    plt.figure()
    plt.title('Histogram of the bottom half of the binary image')
    plt.xlabel('x position (column)')
    plt.ylabel('# of white pixels')
    plt.plot(info['histogram'])
    plt.savefig(os.path.join(OUT_DIR, 'histogram', name + '.png'))
    plt.close()

shutil.copy(os.path.join(OUT_DIR, 'undistort', 'calibration1.jpg'), os.path.join(DOCS_DIR, 'undistort_chessboard.jpg'))
to_docs('undistort', EXAMPLE, 'undistort_road.jpg')
to_docs('threshold', EXAMPLE, 'threshold_example.jpg')
to_docs('perspective', EXAMPLE, 'perspective_example.jpg')
to_docs('lanes', EXAMPLE, 'lanes_example.jpg')
shutil.copy(os.path.join(OUT_DIR, 'histogram', EXAMPLE + '.png'), os.path.join(DOCS_DIR, 'histogram_example.png'))