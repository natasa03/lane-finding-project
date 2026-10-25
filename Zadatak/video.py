import os
import glob
import shutil
import cv2
import numpy as np
import lane_pipeline as lp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'output_videos')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')
EXAMPLE = 'project_video03.mp4'
N_SMOOTH = 5

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


def process_video(in_path, out_path):
    camera = cv2.VideoCapture(in_path)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = camera.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    left_fits = []
    right_fits = []
    frames = 0
    rejected = 0

    while True:
        ret, img = camera.read()
        if not ret:
            break

        h, w = img.shape[:2]
        undist, Minv, left_fit, right_fit = lp.detect_lane(img)

        if left_fit is not None and lp.lane_is_ok(left_fit, right_fit, w, h):
            left_fits = (left_fits + [left_fit])[-N_SMOOTH:]
            right_fits = (right_fits + [right_fit])[-N_SMOOTH:]
        else:
            rejected += 1

        if left_fits:
            result, _ = lp.render_result(undist, Minv, np.mean(left_fits, axis=0), np.mean(right_fits, axis=0))
        else:
            result = undist

        if result.shape[:2] != (height, width):
            result = cv2.resize(result, (width, height))
        writer.write(result)
        frames += 1

    camera.release()
    writer.release()
    return frames, rejected


for path in sorted(glob.glob(os.path.join(BASE_DIR, 'test_videos', '*.mp4'))):
    name = os.path.basename(path)
    out_path = os.path.join(OUT_DIR, name.replace('.mp4', '_result.mp4'))
    frames, rejected = process_video(path, out_path)
    print("%-20s frames: %4d | rejected: %4d (%.1f%%)" % (name, frames, rejected, 100.0 * rejected / max(frames, 1)))

example_result = EXAMPLE.replace('.mp4', '_result.mp4')
shutil.copy(os.path.join(OUT_DIR, example_result), os.path.join(DOCS_DIR, example_result))