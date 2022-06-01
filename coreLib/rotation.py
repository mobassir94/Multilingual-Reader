#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary,Mobassir Hossain
"""
from __future__ import print_function
# ---------------------------------------------------------
# imports
# ---------------------------------------------------------
import numpy as np
import cv2  
from statistics import median_low
from math import floor

def rotate_image(mat, angle):
    """
        Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h),flags=cv2.INTER_NEAREST)
    return rotated_mat

def get_max_width_length_ratio(contour: np.ndarray) -> float:
    """Get the maximum shape ratio of a contour.
    Args:
        contour: the contour from cv2.findContour
    Returns: the maximum shape ratio
    """
    _, (w, h), _ = cv2.minAreaRect(contour)
    return max(w / h, h / w)

def create_mask(image,regions):
    h,w,_=image.shape
    mask=np.zeros((h,w))
    for i, region in enumerate(regions):
        region = np.array(region).astype(np.int32).reshape((-1))
        region = region.reshape(-1, 2)
        cv2.fillPoly(
            mask,
            [region.reshape((-1, 1, 2))],
            255
        )
    return mask
 
def auto_correct_image_orientation(image,result):
    mask=create_mask(image,result)
    h,w=mask.shape
    # extract contours
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours
    contours = sorted(contours, key=get_max_width_length_ratio, reverse=True)

    angles = []
    for contour in contours:
        _, (w, h), angle = cv2.minAreaRect(contour)
        if w / h >  5:  # select only contours with ratio like lines
            angles.append(angle)
        elif w / h < 1 /  5:  # if lines are vertical, substract 90 degree
            angles.append(angle - 90)

    if len(angles) == 0:
        return image,mask,0
    else:
        angle=int(median_low(angles))
        image=rotate_image(image,angle)
        mask=rotate_image(mask,angle)
        return image,mask,angle
         