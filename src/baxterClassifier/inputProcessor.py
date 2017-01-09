from six.moves import xrange
from PIL import Image
import os
import random
import tensorflow as tf
import sys
import cv2
import imutils
import numpy as np
import time
import cPickle

CIFAR_IMG_SIZE = 32
IMAGE_SIZE = 64
CHANNELS = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
NUM_CLASSES = 2


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
            print("cannot read image file : ", filename)
            return None, None, None

    except Exception as e:
        print("=======   EXCEPTION     ======= : ", filename, e)
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

    # img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_RGB = cv2.blur(img_RGB, (3, 3))

    # cv2.imshow("cam", img_RGB)
    # cv2.waitKey(1)
    # time.sleep(3)

    img_resized_np = np.transpose(np.asarray([img_RGB]))
    inputs = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 1), dtype='float32')
    inputs[0] = (img_resized_np / 255.0) * 2.0 - 1.0
    return inputs


def get_next_cifar(batch_size, batch_index):

    # GET BATCH IMAGES FOR TRAINING
    filename = ""
    if batch_index < 10000:
        filename = "data/cifar/data_batch_1"
    elif batch_index < 20000:
        filename = "data/cifar/data_batch_2"
    elif batch_index < 30000:
        filename = "data/cifar/data_batch_3"
    elif batch_index < 40000:
        filename = "data/cifar/data_batch_4"
    else:
        filename = "data/cifar/data_batch_5"

    index = batch_index % 10000

    fo = open(filename, 'rb')
    readData = cPickle.load(fo)
    fo.close()

    images = np.zeros([batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
    annotations = np.zeros([batch_size, 2])
    count = 0
    flag = False

    while count < batch_size:
        img = np.zeros([CIFAR_IMG_SIZE, CIFAR_IMG_SIZE, 3])
        imageData = (readData['data'])[index]

        img[:, :, 0] = (imageData[0:1024]).reshape(
            [CIFAR_IMG_SIZE, CIFAR_IMG_SIZE])
        img[:, :, 1] = (imageData[1024:2048]).reshape(
            [CIFAR_IMG_SIZE, CIFAR_IMG_SIZE])
        img[:, :, 2] = (imageData[2048:3072]).reshape(
            [CIFAR_IMG_SIZE, CIFAR_IMG_SIZE])

        labelData = (readData['labels'])[index]

        if labelData == 0 or labelData == 5:
            if labelData == 5:
                labelData = 1

            label = np.zeros(2)
            label[labelData] = 1

            img_resized = cv2.resize(
                img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

            # SHOW IMAGE
            # cv2.imshow("img", img_resized / 255.0)
            # cv2.waitKey(1)
            # time.sleep(3)

            images[count] = img_resized
            annotations[count] = label
            count += 1

        index += 1
        if index >= 10000:
            index = 0
            flag = True

    skipped = 0
    if flag:
        skipped = batch_size
    else:
        skipped = index - (batch_index % 10000)

    return [images, annotations, skipped]


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def get_sliding_window_img_crops(img_filename):
    annotations = []

    images = []
    boundingBoxInfo = []
    true_img = cv2.imread(img_filename.strip())

    if true_img is None:
        print("no image found in given location....")
        return None

    true_height = true_img.shape[0]
    true_width = true_img.shape[1]

    sizes = [(1, 1), (0.8, 0.8), (0.5, 0.5), (0.3, 0.3)]

    for size in sizes:
        windowSize = (
            int(true_width * size[1]), int(true_height * size[0]))

        for (x, y, window) in sliding_window(true_img, stepSize=32, windowSize=windowSize):
            if window.shape[0] != windowSize[1] or window.shape[1] != windowSize[0]:
                continue

            window = cv2.resize(
                window, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            window = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)

            images.append(window)
            boundingBoxInfo.append([x, y, windowSize[0], windowSize[1]])

    return [true_img, np.array(images), boundingBoxInfo]


def pretrain_read_next(csvFileName, batchSize, batchIndex):
    ins = open(csvFileName)
    lines = ins.readlines()

    with open(csvFileName, 'r') as source:
        readData = [(random.random(), line) for line in source]

    readData.sort()
    readData = readData[0:batchSize + 30]

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

    return [np.array(images), np.array(annotations)]


def getImage(filename, ymin, ymax, xmin, xmax):

    try:
        img = cv2.imread(filename.strip())

        if img is not None:
            crop_img = img[int(xmin):int(xmax), int(ymin):int(ymax)]

        else:
            print("\n\n\n\ncannot read image file : ", filename, " \n\n\n")
            return None

    except Exception as e:
        print("=======   EXCEPTION  ======= : ", filename, e)
        return None

    img_RGB = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    image = np.asarray(img_RGB)

    return image


def augmentImage(image_batch, labels, image, label, index):
    for angle in np.arange(0, 360, 45):

        print("rotating...")
        rotated = imutils.rotate_bound(image, angle)
        img = cv2.resize(
            rotated, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        print(index)
        image_batch[index] = img
        labels[index] = label
        index += 1

        print(index)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        image_batch[index] = img
        labels[index] = label
        index += 1

    return image_batch, labels, index


###
# TODO :
# 1) parse data csv file
# 2) randomly get batch size amount of images
# 3) augment each image data (rotation / flip / parity / blur..)
# 4) put all images into np array
###
def get_imagenet_batch(filename, batchsize):

    with open(filename, 'r') as source:
        readData = [(random.random(), line) for line in source]

    readData.sort()
    data = readData[0:batchsize + 50]
    images = np.zeros([batchsize * 8, IMAGE_SIZE, IMAGE_SIZE, 3])
    labels = np.zeros([batchsize * 8])
    index = 0

    for info in data:
        line = info[1]
        image_data = line.split(",")
        class_label = image_data[0]
        ymin = image_data[1]
        ymax = image_data[2]
        xmin = image_data[3]
        xmax = image_data[4]
        filename = image_data[5]

        img = getImage(filename, ymin, ymax, xmin, xmax)
        if img is not None:
            try:
                images, labels, index = augmentImage(
                    images, labels, img, int(class_label), index)

            except Exception as e:
                print("EXCEPTION : ", e)
                continue

        if index >= batchsize * 8:
            break

    return images, labels


if __name__ == '__main__':
    get_imagenet_batch("data/train_data.csv", 50)
