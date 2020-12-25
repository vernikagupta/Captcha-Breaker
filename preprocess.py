# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 18:24:44 2020

@author: vernika
"""

import cv2
import imutils


def preprocess(image, target_w, target_h):
    # grab the dimensions of the image, then initialize
    # the padding values
#    dim  = None
#    image = cv2.imread(image)
    h,w = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if target_w is None and target_h is None:
        return image

    # if the width is greater than the height
    # then resize along width
    elif w > h:
        image = imutils.resize(image, width=target_w)

    else:
        image = imutils.resize(image, height=target_h)

#    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # however the opposite dimension could be shorter than required fixed size
    # to fix this we will pad the image along shorter dimension
    padW = int((target_w - image.shape[1]) / 2.0)
    padH = int((target_h - image.shape[0]) / 2.0)
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,cv2.BORDER_REPLICATE)

    # Applying this padding should bring our image to our target width and height
    # however, there may be cases where we are one pixel off in a given dimension
    # so apply one more resizing to handle any
    # rounding issues
    image = cv2.resize(image, (target_w, target_h))
    return image



