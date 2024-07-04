
import cv2
import numpy as np


def adaptive_thresholding_and_invert(image):
    """
    Given an input image, this function applies Adaptive Thresholding and inverts the mask.
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    gray_image_blur = cv2.GaussianBlur(gray_image,(5,5),0)
    adaptive_thresh_image = cv2.adaptiveThreshold(
        gray_image_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5
    )
    
    inverted_mask = cv2.bitwise_not(adaptive_thresh_image)
    
    return inverted_mask

def reduce_noise(image):
    """
    This function applies a mix of noise reduction techniques to an input image.
    """
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
    median_filtered = cv2.medianBlur(gaussian_blur, 3)
    bilateral_filtered = cv2.bilateralFilter(median_filtered, 9, 75, 75)
    kernel = np.ones((3,3),np.uint8)
    morph = cv2.morphologyEx(bilateral_filtered, cv2.MORPH_OPEN, kernel)
    #morph = cv2.morphologyEx(bilateral_filtered, cv2.MORPH_TOPHAT, kernel)
    
    return morph

def reduce_noise_and_threshold(image):
    """
    This function applies noise reduction techniques followed by thresholding to an input image.
    """
    denoised_image = reduce_noise(image)
    _, thresholded_image = cv2.threshold(denoised_image, 150, 255, cv2.THRESH_BINARY)
    
    return thresholded_image