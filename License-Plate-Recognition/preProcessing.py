import cv2
import numpy as np
import math

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_HEIGHT = 9

def preProcessing(imageOriginal):

    imageGrayscale = extractValue(imageOriginal)
    
    imageMaxContrastGrayscale = maxmizeContrast(imageGrayscale)
    
    heigth, width = imageGrayscale.shape
    
    imageBlurred = np.zeros((height, width, 1), np.uint8)
    
    imageBlurred = cv2.GaussianBlur(imageMaxContrastGrayscale,GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    
    imageThresh = cv2.adaptiveThreshold(imageBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY

    return imageGrayscale
