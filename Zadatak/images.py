import os
import glob
import shutil
import cv2
import lane_pipeline as lp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'output_images', 'result')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')
EXAMPLE = 'test2'

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

for path in sorted(glob.glob(os.path.join(BASE_DIR, 'test_images', '*.jpg'))):
    name = os.path.splitext(os.path.basename(path))[0]
    result, info = lp.process_image(cv2.imread(path))
    cv2.imwrite(os.path.join(OUT_DIR, name + '.jpg'), result)

    if info is None:
        print("%-20s lane not found" % name)
    else:
        radius, offset = info
        radius_text = "%7.0f m" % radius if radius < 100000 else "> 100000 m"
        print("%-20s radius: %s | offset: %+.2f m" % (name, radius_text, offset))

shutil.copy(os.path.join(OUT_DIR, EXAMPLE + '.jpg'), os.path.join(DOCS_DIR, 'result_example.jpg'))