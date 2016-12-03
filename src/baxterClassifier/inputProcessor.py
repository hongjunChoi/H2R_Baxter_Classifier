from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from inception.dataset import Dataset
from six.moves import xrange  # pylint: disable=redefined-builtin
from PIL import Image
import os
import random
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import sys
import cv2
import numpy as np
import time

IMAGE_SIZE = 112
CHANNELS = 1
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
NUM_CLASSES = 2
BATCH_SIZE = 5

dataset_path = "/path/to/your/dataset/mnist/"
train_labels_file = "train-labels.csv"


def encode_label(label):
    return int(label)


def read_label_file(file):
    f = open(file, "r")
    filepaths = []
    labels = []
    for line in f:
        filepath, label = line.split(",")
        filepaths.append(filepath)
        labels.append(encode_label(label))
    return filepaths, labels


def encodeImg(filename):
    # image = Image.open(filename)
    # image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
    # img = np.array(image)
    try:
        img = cv2.imread(filename.strip())
        if img is not None:
            height, width, channel = img.shape
            img_resized = cv2.resize(
                img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        else:
            # print("cannot read image file : ", filename)
            return None, None, None

    except Exception as e:
        print("=======       EXCEPTION     ======= : ", filename, e)
        return None, None, None

    img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # GREY SCALE
    # img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    # print(img_RGB.shape)
    img_resized_np = np.asarray(img_RGB)
    inputs = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype='float32')
    inputs[0] = (img_resized_np / 255.0) * 2.0 - 1.0
    return inputs, height, width


def cropEncodeImg(filename, boundingBox):
    try:
        img = cv2.imread(filename.strip())
        if img is not None:
            crop_img = img[int(boundingBox[1]):int(boundingBox[2]), int(
                boundingBox[3]):int(boundingBox[4])]
            img_resized = cv2.resize(
                crop_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

            img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            # cv2.imshow("cam", img_resized)
            # cv2.waitKey(1)
            # time.sleep(3)
        else:
            # print("cannot read image file : ", filename)
            return None

    except Exception as e:
        print("=======       EXCEPTION     ======= : ", filename, e)
        return None

    # cv2.imshow("cam", img_RGB)
    # cv2.waitKey(1)
    # time.sleep(3)

    img_resized_np = np.transpose(np.asarray([img_RGB]))
    inputs = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 1), dtype='float32')
    inputs[0] = (img_resized_np / 255.0) * 2.0 - 1.0
    return inputs


def pretrain_read_next(csvFileName, batchSize, batchIndex):
    ins = open(csvFileName)
    lines = ins.readlines()

    with open(csvFileName, 'r') as source:
        readData = [(random.random(), line) for line in source]

    readData.sort()
    readData = readData[0:batchSize + 100]

    images = np.zeros([batchSize, IMAGE_SIZE, IMAGE_SIZE, 1])
    annotations = np.zeros([batchSize, 2])
    count = 0
    index = 0

    while count < batchSize:
        line = readData[index][1]
        index += 1
        data = line.split(",")
        classLabel = data[0]
        ymin = data[1]
        ymax = data[2]
        xmin = data[3]
        xmax = data[4]
        img_filename = ("data/" + data[5]).strip()
        img = cropEncodeImg(img_filename, data)
        if img is None:
            continue

        if classLabel == "a":
            label = [1, 0]
        elif classLabel == "b":
            label = [0, 1]
        else:
            print("----- WRONG ----")
            label = [0, 1]

        images[count] = img[0]
        annotations[count] = label

        count += 1

    return [images, annotations]


def read_next(csvFileName, batchSize, batchIndex):
    ins = open(csvFileName)
    lines = ins.readlines()
    startIndex = batchIndex * batchSize
    endIndex = (batchIndex + 1) * batchSize
    if endIndex >= len(lines):
        endIndex = len(lines) - 1

    nextLines = lines[startIndex:endIndex + 50]

    images = []
    annotations = []
    count = 0
    index = 0

    while count < batchSize:
        line = nextLines[index]
        index += 1
        data = line.split(",")
        classLabel = data[0]
        ymin = data[1]
        ymax = data[2]
        xmin = data[3]
        xmax = data[4]
        img_filename = ("data/" + data[5]).strip()
        img, height, width = encodeImg(img_filename)

        if img is None:
            continue

        images.append(img)

        if classLabel == "a":
            label = 0
        elif classLabel == "b":
            label = 1
        else:
            print("----- WRONG ----")
            label = 1

        label = [label, float(ymin) / height, float(ymax) /
                 height, float(xmin) / width, float(xmax) / width]
        annotations.append(np.asarray(label))
        count += 1

        # file_contents = tf.read_file(img_filename)
        # img = tf.image.decode_png(file_contents)

    return [np.array(images), np.array(annotations)]


def read_my_file_format(filename_and_label_tensor):

    filename, label = tf.decode_csv(
        filename_and_label_tensor, [[""], [""]], " ")
    file_contents = tf.read_file(filename)
    img = tf.image.decode_png(file_contents)
    return img, label


def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)

    img, label = read_my_file_format(filename_queue)

    img_batch, label_batch = tf.train.shuffle_batch(
        [img, label], batch_size=batch_size)
    return img_batch, label_batch


if __name__ == '__main__':
    main(sys.argv)
