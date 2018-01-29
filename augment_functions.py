import numpy as np
import cv2

# transform x y direction, input original rgb image
def transform_image(image):
    trans_range = 4
    tx = trans_range*(np.random.uniform()-0.5)
    ty = 4*(np.random.uniform()-0.5)
    rows,cols,_ = image.shape
    M = np.float32([[1,0,tx],[0,1,ty]])
    image = cv2.warpAffine(image,M,(cols,rows))
    return image

# Horizontal Flip
def flip_image(image):
    # Horizontal Flip
    image = cv2.flip(image, 1)
    return image

# adjust brightness
def bright_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = v.astype(np.float)
    v *= (np.random.uniform()+0.2)
    v[v>255] = 255
    v[v<0] = 0
    v = v.astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    image_hsv = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return image_hsv