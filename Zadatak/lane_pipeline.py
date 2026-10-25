import os
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REF_SIZE = (1280, 720)

calibration = np.load(os.path.join(BASE_DIR, 'camera_cal', 'calib.npz'))
mtx = calibration['mtx']
dist = calibration['dist']


def undistort(img):
    h, w = img.shape[:2]
    scale = np.array([[w / float(REF_SIZE[0]), 0, 0],
                      [0, h / float(REF_SIZE[1]), 0],
                      [0, 0, 1]])
    K = scale @ mtx
    return cv2.undistort(img, K, dist, None, K)


def sobel_x_binary(channel, thresh=(30, 255), ksize=3):
    blurred = cv2.GaussianBlur(channel, (5, 5), 0)
    sobelX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    absX = cv2.convertScaleAbs(sobelX)
    return cv2.inRange(absX, thresh[0], thresh[1])


def threshold_binary(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    yellow = cv2.inRange(hsv, (15, 80, 100), (35, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 200), (179, 40, 255))
    color_binary = cv2.bitwise_or(yellow, white)

    gradient_binary = sobel_x_binary(hls[:, :, 1])
    s_binary = cv2.inRange(hls[:, :, 2], 100, 255)
    gradient_and_s = cv2.bitwise_and(gradient_binary, s_binary)

    return cv2.bitwise_or(color_binary, gradient_and_s)


SRC_REF = np.float32([[578, 460], [203, 720], [1127, 720], [704, 460]])
DST_REL = np.float32([[0.25, 0.0], [0.25, 1.0], [0.75, 1.0], [0.75, 0.0]])


def get_points(w, h):
    src = SRC_REF * np.float32([w / float(REF_SIZE[0]), h / float(REF_SIZE[1])])
    dst = DST_REL * np.float32([w, h])
    return src, dst


def warper(img, src, dst, flags=cv2.INTER_NEAREST):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=flags)
    return warped


NWINDOWS = 9
MARGIN = 100
MINPIX = 50


def find_lane_lines(binary):
    h, w = binary.shape

    histogram = np.sum(binary[h // 2:, :] > 0, axis=0)
    mid = w // 2
    leftx_current = int(np.argmax(histogram[:mid]))
    rightx_current = int(np.argmax(histogram[mid:]) + mid)

    nonzeroy, nonzerox = binary.nonzero()
    window_height = h // NWINDOWS
    left_inds = []
    right_inds = []
    windows = []

    for window in range(NWINDOWS):
        y_low = h - (window + 1) * window_height
        y_high = h - window * window_height
        windows.append(((leftx_current - MARGIN, y_low), (leftx_current + MARGIN, y_high)))
        windows.append(((rightx_current - MARGIN, y_low), (rightx_current + MARGIN, y_high)))

        good_left = ((nonzeroy >= y_low) & (nonzeroy < y_high) &
                     (nonzerox >= leftx_current - MARGIN) & (nonzerox < leftx_current + MARGIN)).nonzero()[0]
        good_right = ((nonzeroy >= y_low) & (nonzeroy < y_high) &
                      (nonzerox >= rightx_current - MARGIN) & (nonzerox < rightx_current + MARGIN)).nonzero()[0]
        left_inds.append(good_left)
        right_inds.append(good_right)

        if len(good_left) > MINPIX:
            leftx_current = int(np.mean(nonzerox[good_left]))
        if len(good_right) > MINPIX:
            rightx_current = int(np.mean(nonzerox[good_right]))

    left_inds = np.concatenate(left_inds)
    right_inds = np.concatenate(right_inds)
    info = {'histogram': histogram, 'windows': windows,
            'left': (nonzerox[left_inds], nonzeroy[left_inds]),
            'right': (nonzerox[right_inds], nonzeroy[right_inds])}

    if len(left_inds) < MINPIX or len(right_inds) < MINPIX:
        return None, None, info

    left_fit = np.polyfit(nonzeroy[left_inds], nonzerox[left_inds], 2)
    right_fit = np.polyfit(nonzeroy[right_inds], nonzerox[right_inds], 2)
    return left_fit, right_fit, info


def draw_search(binary, left_fit, right_fit, info):
    h = binary.shape[0]
    out_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for corner1, corner2 in info['windows']:
        cv2.rectangle(out_img, corner1, corner2, (0, 255, 0), 2)
    leftx, lefty = info['left']
    rightx, righty = info['right']
    out_img[lefty, leftx] = (0, 0, 255)
    out_img[righty, rightx] = (255, 0, 0)

    ploty = np.linspace(0, h - 1, h)
    for fit in (left_fit, right_fit):
        if fit is not None:
            pts = np.array([np.polyval(fit, ploty), ploty]).T.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out_img, [pts], False, (0, 255, 255), 3)
    return out_img


LANE_WIDTH_M = 3.7
LANE_WIDTH_PX = 620.0
ROAD_LENGTH_M = 30.0
MIN_LANE_WIDTH = 0.39
MAX_LANE_WIDTH = 0.59


def meters_per_pixel(w, h):
    xm = LANE_WIDTH_M / (LANE_WIDTH_PX * w / REF_SIZE[0])
    ym = ROAD_LENGTH_M / h
    return xm, ym


def curvature_radius(fit, y_eval, xm, ym):
    A = xm * fit[0] / ym ** 2
    B = xm * fit[1] / ym
    y = y_eval * ym
    return (1 + (2 * A * y + B) ** 2) ** 1.5 / max(abs(2 * A), 1e-9)


def vehicle_offset(left_fit, right_fit, w, h, xm):
    y = h - 1
    left_x = np.polyval(left_fit, y)
    right_x = np.polyval(right_fit, y)
    lane_center = (left_x + right_x) / 2
    return (w / 2 - lane_center) * xm


def draw_lane(undist, left_fit, right_fit, Minv):
    h, w = undist.shape[:2]
    ploty = np.linspace(0, h - 1, h)
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)

    pts_left = np.array([left_fitx, ploty]).T
    pts_right = np.flipud(np.array([right_fitx, ploty]).T)
    pts = np.vstack((pts_left, pts_right)).astype(np.int32)

    color_warp = np.zeros_like(undist)
    cv2.fillPoly(color_warp, [pts], (0, 255, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


def put_text(img, lines):
    for i, text in enumerate(lines):
        cv2.putText(img, text, (30, 50 + i * 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)


def detect_lane(img):
    h, w = img.shape[:2]
    undist = undistort(img)
    binary = threshold_binary(undist)

    src, dst = get_points(w, h)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = warper(binary, src, dst)

    left_fit, right_fit, _ = find_lane_lines(warped)
    return undist, Minv, left_fit, right_fit


def lane_is_ok(left_fit, right_fit, w, h):
    for y in (0, h // 2, h - 1):
        width = np.polyval(right_fit, y) - np.polyval(left_fit, y)
        if not (MIN_LANE_WIDTH * w < width < MAX_LANE_WIDTH * w):
            return False
    return True


def render_result(undist, Minv, left_fit, right_fit):
    h, w = undist.shape[:2]
    xm, ym = meters_per_pixel(w, h)
    radius = (curvature_radius(left_fit, h - 1, xm, ym) + curvature_radius(right_fit, h - 1, xm, ym)) / 2
    offset = vehicle_offset(left_fit, right_fit, w, h, xm)

    result = draw_lane(undist, left_fit, right_fit, Minv)
    if radius > 5000:
        radius_text = 'Curvature radius: > 5000 m (almost straight)'
    else:
        radius_text = 'Curvature radius: %d m' % radius
    side = 'right' if offset > 0 else 'left'
    put_text(result, [radius_text, 'Vehicle is %.2f m %s of center' % (abs(offset), side)])
    return result, (radius, offset)


def process_image(img):
    undist, Minv, left_fit, right_fit = detect_lane(img)
    if left_fit is None:
        result = undist.copy()
        put_text(result, ['Lane not found'])
        return result, None
    return render_result(undist, Minv, left_fit, right_fit)