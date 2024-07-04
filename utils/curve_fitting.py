
import cv2
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from numpy.polynomial.polynomial import Polynomial

# Define the function to fit the data points
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def poly_func(x, a, b, c):
    return a * x**2 + b * x + c

def fit_curve_to_mask(mask, degree=2):
    """
    Given a binary mask, this function treats the mask as a 2D histogram and tries to fit a curve to the white pixels.
    It returns an image with the fitted curve overlaid on the mask.
    """
    # Find the coordinates of the white pixels
    y_coords, x_coords = np.where(mask > 0)
    
    # Fit the curve to the white pixels
    p = Polynomial.fit(x_coords, y_coords, degree)
    
    # Generate the curve using the optimized parameters
    x_curve = np.linspace(0, mask.shape[1]-1, mask.shape[1])
    y_curve = p(x_curve)
    
    # Create an output image to overlay the curve
    output_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Draw the curve on the output image
    for i in range(len(x_curve) - 1):
        pt1 = (int(x_curve[i]), int(y_curve[i]))
        pt2 = (int(x_curve[i + 1]), int(y_curve[i + 1]))
        cv2.line(output_image, pt1, pt2, (0, 255, 0), 1)
    
    return output_image

# Define Huber loss function
def huber_loss(params, x, y, delta=1.0):
    a, b, c = params
    residual = y - (a * x**2 + b * x + c)
    loss = np.where(np.abs(residual) < delta, 0.5 * residual**2, delta * (np.abs(residual) - 0.5 * delta))
    return np.sum(loss)

def fit_curve_with_huber_loss(mask, delta=2.0):
    """
    Given a binary mask, this function treats the mask as a 2D histogram and tries to fit a curve to the white pixels
    using Huber loss. It returns an image with the fitted curve overlaid on the mask.
    """
    # Find the coordinates of the white pixels
    y_coords, x_coords = np.where(mask > 0)
    
    # Initial guess for the polynomial parameters a, b, c in ax^2 + bx + c
    initial_params = [1, 1, 1]
    
    # Optimize the Huber loss
    result = minimize(huber_loss, initial_params, args=(x_coords, y_coords, delta))
    
    # Extract the optimized parameters
    a, b, c = result.x
    
    # Generate the curve using the optimized parameters
    x_curve = np.linspace(0, mask.shape[1] - 1, mask.shape[1])
    y_curve = a * x_curve**2 + b * x_curve + c
    
    # Create an output image to overlay the curve
    mask_with_curve = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    only_curve_mask = np.zeros_like(mask)
    # Draw the curve on the output image
    for i in range(len(x_curve) - 1):
        pt1 = (int(x_curve[i]), int(y_curve[i])+1)
        pt2 = (int(x_curve[i + 1]), int(y_curve[i + 1])+1)
        cv2.line(mask_with_curve, pt1, pt2, (0, 255, 0), 1)
        cv2.line(only_curve_mask, pt1, pt2, 255, 1)
    
    # Find contours on the original mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank output mask
    output_mask = np.zeros_like(mask)

    # For each contour, check if it intersects with the curve. If yes, draw the entire contour on the output mask.
    for contour in contours:
        # Create a blank temporary mask for the current contour
        temp_mask = np.zeros_like(mask)
        cv2.drawContours(temp_mask, [contour], -1, 255, -1)  # Draw filled contour on temporary mask

        # Check intersection between contour and curve
        intersection = cv2.bitwise_and(temp_mask, only_curve_mask)
        
        # If there's an intersection, draw the contour on the output mask
        if np.any(intersection):
            cv2.drawContours(output_mask, [contour], -1, 255, -1)
        
    return mask_with_curve,output_mask