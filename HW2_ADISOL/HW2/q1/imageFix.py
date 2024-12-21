# Yoav Simani, 208774315
# Adi Raz, 206875874

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import matplotlib.pyplot as plt
import numpy as np


def brightness_and_contrast_stretch(img, brightness=20):
    width, height = img.shape

    # find min and max gray values
    min_intensity = min(img.flatten())
    max_intensity = max(img.flatten())

    # change the contrast and the brightness of the image
    for x in range(width):
        for y in range(height):
            intensity = img[x, y]
            norm = (intensity - min_intensity) / (max_intensity - min_intensity)
            updated_intensity = 255 * norm
            updated_intensity += brightness
            img[x, y] = max(0, min(int(updated_intensity), 255))

    return img


def gamma_correction(img, gamma=1.0):
    normalized_image = img / 255.0
    img_after_gamma = (np.power(normalized_image, gamma)) * 255.0
    return img_after_gamma


def histogram_equalization(img):
    return cv2.equalizeHist(img)


def apply_fix(image, id):
    if id == 1:
        img = histogram_equalization(image)
    elif id == 2:
        img = gamma_correction(image, 1/2.2)
    elif id == 3:
        img = brightness_and_contrast_stretch(image, -20)  # doesn't change much but still makes it a little bit better

    return img


for i in range(2, 4):
    if (i == 1):
        path = f'{i}.png'
    else:
        path = f'{i}.jpg'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    fixed_image = apply_fix(image, i)
    if (i == 1):
        plt.imsave(f'{i}_fixed.png', fixed_image, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imsave(f'{i}_fixed.jpg', fixed_image, cmap='gray', vmin=0, vmax=255)
