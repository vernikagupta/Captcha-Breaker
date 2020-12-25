# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:04:14 2020

@author: vernika
"""

import cv2
import argparse
import glob
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
help="path to input directory of images")
ap.add_argument("-a", "--annot", required=True,
help="path to output directory of annotations")
args = vars(ap.parse_args())


imagePaths = [f for f in glob.glob(args["input"] + "\\" + "*.jpg")]
counts = {}

for i, imagepath in enumerate(imagePaths):
    # print("processing image {}/{}".format((i+1),len(imagePaths)))

    # load the image and convert it to grayscale, then pad the
    # image to ensure digits caught on the border of the image
    # are retained
    image = cv2.imread(imagepath)
    # cv2.imshow("image",image)
    # cv2.waitKey(0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    # threshold the image to reveal the digits
    thresh =  cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cnts,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find contours in the image, keeping only the four largest
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]


    # loop over the contours
    for cnt in cnts:
        # compute the bounding box for the contour then extract
        # the digit
        (x, y, w, h) = cv2.boundingRect(cnt)
        roi = gray[y-5 : y+h+5, x-5 : x+w+5 ]
        # display the character, making it larger enough for us
        # to see, then wait for a keypress
        cv2.imshow("roi", roi)
        key = cv2.waitKey(0)

        # if the ’‘’ key is pressed, then ignore the character
        # Needing to ignore a character may happen if our script
        # accidentally detects “noise” (i.e., anything but a digit)
        if key == "'":
            print("[INFO] ignoring character")
            continue

        # grab the key that was pressed and construct the path
        # the output directory
        key = chr(key).upper()
        dirpath = os.path.sep.join([args["annot"], key]) # os.path.sep gives string object rom given object

        # if the output directory does not exist, create it
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        # write the labeled character to file
        count = counts.get(key, 1)
        p = os.path.sep.join([dirpath, "{}.png".format(str(count).zfill(6))])
        cv2.imwrite(p, roi)

        # increment the count for the current key
        counts[key] = count + 1








