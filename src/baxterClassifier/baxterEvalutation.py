import os
import glob
import sys
import skimage
import numpy as np
import cv2
import time
import cPickle
import math
import inputProcessor
import xml.etree.ElementTree as ET
import copy
import baxterDetector as detector
import matplotlib.pyplot as plt


def calculatePrecision(label, detectedBoxList, annotatedData):
    # TODO: over all detected box in detectedBoxList, get me the ones that are actually correct
    # that matches box in annotatedData
    groundTruth = copy.deepcopy(annotatedData)
    detectedList = copy.deepcopy(detectedBoxList)

    total = 0
    count = 0
    for detection in detectedBoxList:
        if detection['category'] == label:
            total = total + 1

    if total == 0:
        print("no bounding box detected due to high threshold...")
        return 0

    for d in detectedList:
        for g in groundTruth:
            box0 = g['box']
            class0 = g['category']

            box1 = d['box']
            class1 = d['category']

            if class0 == class1 or class0 == None:
                if inputProcessor.intersection_over_union(box0, box1) > 0.5:
                    count = count + 1
                    groundTruth.remove(g)
                    break

    return float(count) / float(total)


def calculateRecall(label, detectedBoxList, annotatedData):
    # TODO: over all ground truth annotated data, get me the ones that are
    # actually detected.
    groundTruth = copy.deepcopy(annotatedData)
    detectedList = copy.deepcopy(detectedBoxList)

    total = 0
    count = 0

    for truth in annotatedData:
        if truth['category'] == label:
            total = total + 1

    if total == 0:
        print("returning 1 since no box was annotated (no ground truth in image) ..")
        return 1

    for g in groundTruth:
        for d in detectedList:
            box0 = g['box']
            class0 = g['category']

            box1 = d['box']
            class1 = d['category']

            if class0 == class1 or class0 is None:
                if inputProcessor.intersection_over_union(box0, box1) > 0.5:
                    count = count + 1
                    detectedList.remove(d)
                    break

    return float(count) / float(total)


def parseAnnotatedData(xmlFilePath):
    # returns box data consistent with detection along with the category
    tree = ET.parse(xmlFilePath)
    root = tree.getroot()
    boxData = []
    for boundingBoxObject in root.findall('object'):
        box = []
        category = convertCategory2Label(boundingBoxObject.find('name').text)

        polygon = boundingBoxObject.find('polygon')
        points = polygon.findall('pt')
        x = int(points[0].find('x').text)
        y = int(points[0].find('y').text)
        w = (int(points[1].find('x').text) -
             int(points[0].find('x').text))
        h = (int(points[3].find('y').text) -
             int(points[0].find('y').text))

        box = [x, y, w, h]
        boxData.append({'category': category,
                        'box': box})

    return boxData


def convertCategory2Label(category):
    if category == "spoon":
        return 0
    else:
        return 1


def getAllTestImages(path):
    return


def plotRegionPRCurve():
    return


def validateAnnotation(testImagePath, annotationPath):
    image = skimage.io.imread(testImagePath)
    annotatedData = parseAnnotatedData(annotationPath)

    for true_box in annotatedData:
        box = true_box['box']

        cv2.rectangle(image, (box[0], box[1]),
                      (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 1)

    cv2.imshow("Window", image)
    cv2.waitKey(1000)
    time.sleep(10)


def plotDetectionPRCurve(groundTruthPath, testImagePath, label):
    threshold_list = [0.1, 0.5, 0.7, 1, 1.3, 1.7, 2,
                      2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 11, 12]
    precision_list = []
    recall_list = []
    baxterDetector = detector.BaxterDetector()

    for threshold in threshold_list:
        baxterDetector.threshold = threshold
        boundingBoxData = baxterDetector.detectObject(label, testImagePath)
        annotatedData = parseAnnotatedData(groundTruthPath)

        precision = calculatePrecision(label, boundingBoxData, annotatedData)
        recall = calculateRecall(label, boundingBoxData, annotatedData)
        precision_list.append(precision)
        recall_list.append(recall)

    precision_list.insert(0, 1)
    recall_list.insert(0, 0)

    precision_list = [x for (y, x) in sorted(
        zip(recall_list, precision_list))]
    recall_list.sort()

    plot_PR_curve("PR Curve", recall_list, precision_list,
                  "recall", "precision", "precision-recall values")
    return


def plot_PR_curve(title, recall_x, precision_y, x_label, y_label, label, y_min=0, y_max=1):
    plt.figure()
    plt.title(title)
    ylim = (y_min, y_max)
    plt.ylim(*ylim)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()

    plt.plot(recall_x, precision_y, 'o-', color="r",
             label=label)

    plt.legend(loc="best")
    plt.show()
    return


def plot_region_proposal_recall_curve():
    test_image_path = "data/synthetic_test/*.jpg"
    test_annotation_path = "data/synthetic_test_annotations/"
    test_images = glob.glob(test_image_path)
    average_recall_list = []
    tunable_param_list = [0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 4, 5, 10]
    recall = 0

    for param in tunable_param_list:
        print("adding .. " + str(float(recall) / float(len(test_image_path))))
        average_recall_list.append(float(recall) / float(len(test_image_path)))
        recall = 0

        for image_path in test_images:
            annotation_path = test_annotation_path + \
                (image_path.split("/")[2]).split('.')[0] + ".xml"
            ground_truth = parseAnnotatedData(annotation_path)

            for i in range(len(ground_truth)):
                ground_truth[i]['category'] = None

            detected_data = []
            proposals = inputProcessor.regionProposal(
                image_path, 64, sigma=param)
            boxData = proposals[2]

            for box in boxData:
                data = {}
                data['category'] = None
                data['box'] = box
                detected_data.append(data)
            recall = recall + \
                calculateRecall(None, detected_data, ground_truth)

    print("-------   getting region proposal recall value ---------")
    print(tunable_param_list)
    print(average_recall_list)
    return


def plot_region_proposal_results():
    sigma = [0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 4, 5, 10]
    recall_vals = [0.0, 0.822, 0.8175, 0.825,
                   0.802, 0.7475, 0.6715, 0.666, 0.565, 0.585]

    plot_PR_curve("Recall Curve for Region Proposal Algorithm", sigma, recall_vals,
                  "Sigma", "Recall Value", "Recall")

if __name__ == '__main__':
    # plot_region_proposal_results()
    # plot_region_proposal_recall_curve()

    plotDetectionPRCurve(
        "data/synthetic_test_annotations/test18.xml", "data/synthetic_test/test18.jpg", 1)
