import cv2
import numpy as np

import proc

count = 0
pts = []
flag = False

def mouse_callback(event, x, y, flags, param):
    global count, pts, flag    
    if event == cv2.EVENT_LBUTTONDOWN:
        if count < 4 and not flag:
            pts.append((x, y))
            count += 1
            
        flag = True
    elif event == cv2.EVENT_LBUTTONUP:
        flag = False
    
    if count == 4:
        cv2.destroyAllWindows()

def preprocess_image(image, param=None):
    if param is not None:
        param = {**create_random_params(), **param}
    
    image = proc.add_blur(image, ksize=param["blur_ksize"])
    image = proc.adjust_color(image, brightness=param["brightness"], color_shift=param["color_shift"])
    image = proc.add_distortion(image, strength=param["distortion_strength"])
    image = proc.add_noise(image, amount=param["noise_amount"])
    image = proc.jpeg_compress(image, quality=param["jpeg_quality"])
    
    return image

def create_random_params():
    blur_ksize = 2 * np.random.randint(1, 10) + 1
    brightness = np.random.uniform(0.6, 1.4)
    color_shift = (np.random.randint(-5, 5), np.random.randint(-5, 5), np.random.randint(-5, 5))
    distortion_strength = np.random.uniform(0.0001, 0.001)
    noise_amount = np.random.uniform(0.01, 0.1)
    jpeg_quality = np.random.randint(30, 50)
    
    param = {
        "blur_ksize": blur_ksize,
        "brightness": brightness,
        "color_shift": color_shift,
        "distortion_strength": distortion_strength,
        "noise_amount": noise_amount,
        "jpeg_quality": jpeg_quality}
    
    return param

if __name__ == "__main__":

    source_img = cv2.imread("NineGridTransp.png", cv2.IMREAD_UNCHANGED)
    source_img = source_img.copy()[:1772,:,:]
    
    transparent_mask = source_img[:, :, 3] == 0
    param = create_random_params()
    source_img = preprocess_image(source_img[:, :, :3], param=param)
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2BGRA)
    source_img[transparent_mask] = [0, 0, 0, 0]    

    cv2.namedWindow("Image with Transparency", cv2.WINDOW_NORMAL)
    cv2.imshow("Image with Transparency", source_img)
    

    target_img = cv2.imread("test_image1.jpg", -1)
    cv2.namedWindow("target_img", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("target_img", mouse_callback, param=target_img)
    cv2.imshow("target_img", target_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    source_pts = np.array([[0, 0], 
                            [source_img.shape[1]-1, 0], 
                            [source_img.shape[1]-1, source_img.shape[0]-1], 
                            [0, source_img.shape[0]-1]], dtype=np.float32)
    target_pts = np.array(pts, dtype=np.float32)

    homography_matrix, _ = cv2.findHomography(source_pts, target_pts)
    warped_img = cv2.warpPerspective(source_img, homography_matrix, (target_img.shape[1], target_img.shape[0]))
    alpha_channel = warped_img[:, :, 3] / 255.0

    for c in range(0, 3):
        target_img[:, :, c] = (alpha_channel * warped_img[:, :, c] + (1 - alpha_channel) * target_img[:, :, c])
    cv2.namedWindow("Transformed Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Transformed Image", target_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.imwrite("transformed_image.png", target_img)