import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def best_fit_line_and_angle(image):
    """
    Given a 2D binary mask, this function calculates the best fit line for all the white pixels
    and draws it on the mask. It also calculates the angle of the line with respect to the x-axis.
    
    Returns:
    - Image with the best fit line drawn
    - Angle of the line
    """
    # Get the coordinates of the white pixels
    y, x = np.where(image == 255)
    
    # Fit a line using np.polyfit with degree 1 (linear fit)
    coefficients = np.polyfit(x, y, 1)
    slope, intercept = coefficients
    
    # Calculate the endpoints of the best fit line within the image dimensions
    x1 = 0
    y1 = int(intercept)
    x2 = image.shape[1] - 1
    y2 = int(slope * x2 + intercept)
    
    # Create a copy of the input image to draw the best fit line
    output_image = np.dstack([image] * 3)
    
    # Draw the best fit line on the image
    cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Calculate the angle of the line with respect to the x-axis
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    
    return output_image, angle

# Function to remove the most top-left and most top-right contours
def remove_extreme_contours(image):
    """
    Given a binary image, this function removes the most top-left and most top-right contours.
    """
    # Find the contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If there are fewer than 2 contours, return the original image
    if len(contours) < 2:
        return image.copy()
    
    # Calculate the centroids of the contours
    centroids = [np.mean(contour, axis=0) for contour in contours]
    centroids = np.array([c[0] for c in centroids])
    
    # Sort the centroids by their x-coordinates to find the most left and most right contours
    sorted_indices = np.argsort(centroids[:, 0])
    most_left_contour = contours[sorted_indices[0]]
    most_right_contour = contours[sorted_indices[-1]]
    
    # Create a mask to remove the most left and most right contours
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [most_left_contour, most_right_contour], -1, 255, -1)
    
    # Subtract the mask from the original image to remove the contours
    result_image = cv2.subtract(image, mask)
    
    return result_image

def mean_area_contours(contours):
    # Calculate the area for each contour
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    
    # Calculate the median of the contour areas
    mean_area = np.mean(contour_areas)
    return contour_areas,mean_area

def visualize_perfect_rectangle(contour, image_shape):
    """
    Given a single contour and image shape, this function visualizes a perfect rectangle
    with the longest side of the contour.
    
    Returns:
    - Image with the original contour (Red)
    - Image with the perfect rectangle (Green)
    """
    # Create empty images to draw the original contour and the perfect rectangle
    contour_image = np.zeros(image_shape, dtype=np.uint8)
    rectangle_image = np.zeros(image_shape, dtype=np.uint8)
    
    # Draw the original contour on the image (Red)
    cv2.drawContours(contour_image, [contour], -1, (0, 0, 255), -1)
    
    # Get the center of the contour
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    # Get the dimensions of the minimum area rectangle around the contour
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    
    # Create the perfect rectangle with the longest side of the contour
    side_length = max(width, height)
    x1 = int(cX - side_length // 2)
    y1 = int(cY - side_length // 2)
    x2 = int(cX + side_length // 2)
    y2 = int(cY + side_length // 2)
    
    # Draw the perfect rectangle on the image (Green)
    cv2.rectangle(rectangle_image, (x1, y1), (x2, y2), (0, 255, 0), -1)
    
    return contour_image, rectangle_image


def plot_contour_area_histogram(contours):
    """
    Given a list of contours, this function plots a histogram of contour area sizes.
    It also determines and plots the median of contour areas as a red line.
    """
    if not contours:
        return
    # Calculate the area for each contour
    contour_areas,mean_area = mean_area_contours(contours=contours)
    

    # Create a histogram plot
    plt.figure(figsize=(10, 5))
    plt.hist(contour_areas, bins=np.linspace(min(contour_areas), max(contour_areas), 20), color='blue', edgecolor='black')
    plt.axvline(mean_area, color='red', linestyle='dashed', linewidth=2)
    plt.title('Histogram of Contour Area Sizes')
    plt.xlabel('Contour Area')
    plt.ylabel('Frequency')
    plt.grid(True)