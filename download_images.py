# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 13:04:01 2020

@author: vernika
"""

import argparse
import requests
import os
import time

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to output directory of images')
ap.add_argument('-n', '--num_images', type=int, required=True, help="# of images to download")
args = vars(ap.parse_args())

# initialize the URL that contains the captcha images that we will
# be downloading along with the total number of images downloaded
# thus far
url = "https://www.e-zpassny.com/vector/jcaptcha.do"
total = 0

# loop over the number of images to download
for no in range(0,args["num_images"]):
    try:
        r = requests.get(url, timeout=60)

        # save the image to disk
        p = os.path.sep.join([args['output'], "{}.jpg".format(str(total).zfill(5))])
        f = open(p, 'w')
        f.write(r.content)
        f.close()

        # update the counter
        print("[INFO] total images downloaded {}".format(total))
        total += 1

    # handle if any exceptions are thrown during the download process
    except:
        print("Error downloading images")

    # insert a small sleep to be courteous to the server
    time.sleep(0.2)

