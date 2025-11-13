import cv2
import numpy as np

# Gaussian blur function
def add_blur(image, ksize=3):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

# Noise addition function
def add_noise(image:np.ndarray, amount=0.02):
    noise = np.random.normal(0, 255 * amount, image.shape).astype(np.int16)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy

# Color adjustment function
def adjust_color(image, brightness=0.95, color_shift=(5, -3, 0)):
    img = image.astype(np.float32) * brightness
    img[..., 0] += color_shift[0]  # R
    img[..., 1] += color_shift[1]  # G
    img[..., 2] += color_shift[2]  # B
    return np.clip(img, 0, 255).astype(np.uint8)

# Distortion function
def add_distortion(image, strength=0.0005):
    h, w = image.shape[:2]
    K = np.array([[w, 0, w / 2],
                  [0, w, h / 2],
                  [0, 0, 1]])
    D = np.array([strength, strength, 0, 0])
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), 5)
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

# JPEG compression function
def jpeg_compress(image, quality=40):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)