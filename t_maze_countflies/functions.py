
# Imports
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


def countFlies(inputImage):
    '''
    This is the main function that will be called by the main code to count flies given in an image

    The steps are as follows:
        - process image
        - count objects
        - returns an integer (number of flies)
    '''

    processedImage = imagePrepare(inputImage)
    numOfFlies = objCount(processedImage)

    return numOfFlies


def imagePrepare(imgPath):
    '''
    This function will take an input image as an argument and will perform all the processing before the counting of objects
    It will:
        - Threshold
        - Reduce noise
        - Remove artifacts
        - Identify FOV (maybe, need to think about it)
    '''
    # Load the image
    image = Image.open(imgPath)

    # Convert image to grayscale
    gray_image = image.convert('L')

    # Convert the grayscale image to a numpy array
    gray_array = np.array(gray_image)

   # Apply simple thresholding
    threshold_value = 128
    _, threshold_array = cv2.threshold(gray_array, threshold_value, 255, cv2.THRESH_BINARY)

    # Perform Canny edge detection on the thresholded image
    edges = cv2.Canny(threshold_array, threshold1=150, threshold2=160)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw the filtered contours
    mask = np.zeros_like(edges)

    # Filter contours by area
    min_area = 3  # Adjust this value as needed to remove small objects
    max_area = 40
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Display the original, thresholded, edge-detected, and filtered edge-detected images side by side
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Thresholded image
    axes[1].imshow(threshold_array, cmap='gray')
    axes[1].set_title('Thresholded Image')
    axes[1].axis('off')

    # Edge-detected image
    axes[2].imshow(edges, cmap='gray')
    axes[2].set_title('Canny Edge Detection')
    axes[2].axis('off')

    # Filtered edge-detected image
    axes[3].imshow(mask, cmap='gray')
    axes[3].set_title('Filtered Canny Edge Detection')
    axes[3].axis('off')

    plt.show()

def objCount():
    '''
    This function will count objects in a given image
    It will:
        - segment between objects
        - assign minimum & maximum size
        - return an integer (number of objects)
    '''
    num = 0

    return num