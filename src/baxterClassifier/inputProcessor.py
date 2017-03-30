from six.moves import xrange
from PIL import Image
import skimage
import selectivesearch
import os
import random
import glob
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
        else:
            return None

    except Exception as e:
        print("=======       EXCEPTION     ======= : ", filename, e)
        return None

    # img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_RGB = cv2.blur(img_RGB, (3, 3))

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


def getCroppedImage(filename, ymin, ymax, xmin, xmax):

    try:
        img = cv2.imread(filename.strip())

        if img is not None:
            crop_img = img[int(xmin):int(xmax), int(ymin):int(ymax)]

        else:
            return None

    except Exception as e:
        print("=======   EXCEPTION  ======= : ", filename, e)
        return None

    img_RGB = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    image = np.asarray(img_RGB)

    return image


def getImage(filename):

    try:
        img = cv2.imread(filename.strip())

        if img is None:
            return None

    except Exception as e:
        print("=======   EXCEPTION  ======= : ", filename, e)
        return None

    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = np.asarray(img_RGB)

    return image


def augmentImage(image_batch, labels, image, label, index):
    for angle in np.arange(0, 360, 90):
        rotated = imutils.rotate_bound(image, angle)
        img = cv2.resize(
            rotated, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        image_batch[index] = img
        cv2.imshow("cam", img)
        cv2.waitKey(1000)
        time.sleep(1)

        if label == 0:
            labels[index] = [1, 0]
        else:
            labels[index] = [0, 1]

        index += 1

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        image_batch[index] = img
        cv2.imshow("cam", img)
        cv2.waitKey(1000)
        time.sleep(1)

        if label == 0:
            labels[index] = [1, 0]
        else:
            labels[index] = [0, 1]
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
    labels = np.zeros([batchsize * 8, 2])
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

        img = getCroppedImage(filename, ymin, ymax, xmin, xmax)
        if img is not None:
            try:
                images, labels, index = augmentImage(
                    images, labels, img, int(class_label), index)

            except Exception as e:
                print("EXCEPTION : ", e)
                continue

        if index >= batchsize * 8:
            break

    c = list(zip(images, labels))
    random.shuffle(c)
    a, b = zip(*c)
    return [a, b]


def get_caltech_dataset_batch(batch_size):
    # TODO: change the file path as needed
    class1_path = "data/laptop_caltech/*.jpg"
    class2_path = "data/umbrella_caltech/*.jpg"

    class1Data = [(random.random(), data) for data in glob.glob(class1_path)]
    class2Data = [(random.random(), data) for data in glob.glob(class2_path)]

    class1Data.sort()
    class2Data.sort()

    half = batch_size / 2

    class1Data = class1Data[0:half]
    class2Data = class2Data[0:half]

    images = class1Data + class2Data

    image_batch = np.zeros([batch_size * 8, IMAGE_SIZE, IMAGE_SIZE, 3])
    label_batch = np.zeros([batch_size * 8, 2])
    index = 0

    for i in range(len(images)):
        if i < batch_size / 2:
            label = 0
        else:
            label = 1

        path = images[i][1]
        image = getImage(path)
        image_batch, label_batch, index = augmentImage(
            image_batch, label_batch, image, label, index)

    c = list(zip(image_batch, label_batch))
    random.shuffle(c)
    a, b = zip(*c)

    return [a, b]


def getNormalizationData(trainingDataPath):

    with open(trainingDataPath, 'r') as f:
        num_lines = sum(1 for line in f)
        imageList = np.zeros([num_lines, IMAGE_SIZE, IMAGE_SIZE, 3])

    with open(trainingDataPath, 'r') as source:
        index = 0

        for line in source:
            path = line.split(",")[0]
            image = getImage(path)

            if image is None:
                continue

            img_resized = cv2.resize(
                image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

            imageList[index] = img_resized
            index = index + 1

        meanImage = np.mean(imageList, axis=0)
        std = np.std(imageList, axis=0)
        return [meanImage, std]


def preprocessImage(image, meanImage, std):
    return (image - meanImage) / std


def get_custom_dataset_batch(batch_size, train_dataset_path, meanImage, std):
    image_batch = np.zeros([batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
    label_batch = np.zeros([batch_size, 2])
    with open(train_dataset_path, 'r') as source:
        data = [(random.random(), line) for line in source]
        data.sort()

        count = 0
        index = 0

        while(count < batch_size):
            line = data[index][1].split(",")
            path = str(line[0])
            classLabel = int(line[1])
            image = getImage(path)

            if image is None:
                index = index + 1
                continue

            img_resized = cv2.resize(
                image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

            if classLabel is 0:
                label = [1, 0]
            else:
                label = [0, 1]

            processed_image = preprocessImage(img_resized, meanImage, std)

            image_batch[count] = processed_image
            label_batch[count] = label
            index = index + 1
            count = count + 1

    return [image_batch, label_batch]


def get_sliding_window_img_crops(img_filename):
    annotations = []

    images = []
    boundingBoxInfo = []
    true_img = cv2.imread(img_filename.strip())

    if true_img is None:
        print("no image found in given location....")
        return None

    true_img = cv2.resize(
        true_img, (300, 300), interpolation=cv2.INTER_AREA)

    true_height = true_img.shape[0]
    true_width = true_img.shape[1]

    sizes = [(1, 1), (0.8, 0.8), (0.5, 0.5), (0.3, 0.3)]

    for size in sizes:
        windowSize = (
            int(true_width * size[1]), int(true_height * size[0]))

        for (x, y, window) in sliding_window(true_img, stepSize=32, windowSize=windowSize):
            if window.shape[0] != windowSize[1] or window.shape[1] != windowSize[0]:
                continue
            window = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
            window = cv2.resize(
                window, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

            images.append(window)
            boundingBoxInfo.append([x, y, windowSize[0], windowSize[1]])

    return [true_img, np.array(images), boundingBoxInfo]


def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def regionProposal(image_filename, size):
    true_image = skimage.io.imread(image_filename)
    if true_image is None:
        print("CANNOT FIND IMAGE TO CONDUCT DETECTION")
        return None

    img = cv2.resize(
        true_image, (300, int(300.0 * float(true_image.shape[0]) / float(true_image.shape[1]))), interpolation=cv2.INTER_AREA)

    img_size = true_image.shape
    boundingBoxInfo = []
    images = []
    aspect_ratio = float(img_size[1]) / 300.0

    img_lbl, regions = selectivesearch.selective_search(
        img, scale=250, sigma=1)

    candidates = set()

    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # # # excluding regions smaller than 400 pixels
        if r['size'] < 400:
            continue
        # # distorted rects
        x, y, w, h = r['rect']
        if (w > 280 or h > 280) or (w < 15 or h < 15):
            continue

        candidates.add(r['rect'])

    for x, y, w, h in candidates:

        window = img[y: y + h, x: x + w]

        window = cv2.resize(
            window, (size, size), interpolation=cv2.INTER_AREA)

        images.append(window)

        new_x = int(x * aspect_ratio)
        new_y = int(y * aspect_ratio)
        new_w = int(w * aspect_ratio)
        new_h = int(h * aspect_ratio)

        boundingBoxInfo.append([new_x, new_y, new_w, new_h])

        cv2.rectangle(true_image, (new_x, new_y),
                      (new_x + new_w, new_y + new_h), (0, 0, 255), 1)

    # cv2.imshow("Window", true_image)
    # cv2.waitKey(1000)
    # time.sleep(10)

    return [true_image, np.array(images), boundingBoxInfo]


if __name__ == '__main__':
    regionProposal('data/synthetic_test/both1.png', 200)
    # regionProposal('data/test_custom/both5.jpg', 200)

    # regionProposal('data/fromJohn/testImages/spoon2.png')
    # [image_batch, label_batch] = get_custom_dataset_batch(
    #     25, "data/custom_train_data.csv")

    # print(image_batch.shape)
    # print(label_batch.shape)
    # print(label_batch
