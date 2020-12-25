# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 19:49:04 2020

@author: vernika
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from cnn.lenet import LeNet
from cnn.minivggnet import MiniVGGNet
import preprocess
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from imutils import paths


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
args = vars(ap.parse_args())

# initialize the data and labels
data =[]
labels = []

# loop over the input images
for imagePath in paths.list_images(args["dataset"]):
    # load the image, pre-process it, and store it in the data list
#    path = args['dataset'] + "\\" + img
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess.preprocess(image,28,28)
    # converting image to keras compatibility
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # label list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype=float) / 255.0
labels = np.array(labels)


# Partition the data into training data (75%) and testing data (25%)
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
lb = LabelBinarizer().fit(train_y)
train_y = lb.transform(train_y)
test_y = lb.transform(test_y)

# initialize the model
print("Compiling model")
lenet = LeNet.build(width=28, height=28, depth=1, classes=9)
opt = SGD(lr=0.01)
lenet.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
H = lenet.fit(train_x, train_y, validation_data=(test_x,test_y),
              batch_size=16, epochs=5, verbose=1)

# Evaluate the network
print("[INFO]: Evaluating....")
predictions = lenet.predict(test_x, batch_size=16)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# Save the model to disk
print("[INFO]: Serializing network....")
lenet.save(args["model"])

# Plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 5), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 5), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 5), H.history["accuracy"], label="acc")
plt.plot(np.arange(0, 5), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()