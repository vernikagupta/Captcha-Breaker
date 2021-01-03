from keras.models import load_model
import preprocess as p
from imutils import contours
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array


class captcha_breaker:
    def __init__(self, imagename):
        self.imagename = imagename

    def prediction(self):
        image_path = self.imagename
        model = load_model(r'C:\Users\verni\Desktop\Pyimagesearch\Captcha-Breaker\model\lenet.hdf5')
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        # Threshold the image to reveal the digits
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Find contours in the image, keeping only the four largest ones, then sort them from left-to-right
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
        cnts = contours.sort_contours(cnts)[0]

        # Initialize the output image as a "grayscale" image with 3 channels along with the output predictions
        output = cv2.merge([gray] * 3)
        predictions = []
        # Loop over the contours
        for c in cnts:
            # Compute the bounding box for the contour then extract the digit
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

            # Pre-process the ROI and classify it
            roi = p.preprocess(roi, 28, 28)
            roi = np.expand_dims(img_to_array(roi), axis=0)
            pred = model.predict(roi).argmax(axis=1)[0] + 1
            predictions.append(str(pred))
        final_pred = "".join(predictions)
        return [{ "image" : final_pred}]

