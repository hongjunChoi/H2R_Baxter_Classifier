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
        return 1

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

        x_list = [int(points[0].find('x').text), int(points[1].find('x').text), int(
            points[2].find('x').text), int(points[3].find('x').text)]
        y_list = [int(points[0].find('y').text), int(points[1].find('y').text), int(
            points[2].find('y').text), int(points[3].find('y').text)]

        x_min = min(x_list)
        y_min = min(y_list)
        x_max = max(x_list)
        y_max = max(y_list)

        w = x_max - x_min
        h = y_max - y_min

        box = [x_min, y_min, w, h]
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


def exportAnnotationData(testImageFolderPath):
    test_image_path_list = glob.glob(testImageFolderPath + "*.jpg")
    annotation_data = []

    for i in range(len(test_image_path_list)):
        test_image_path = test_image_path_list[i]

        temp = test_image_path.split('/')
        annotation_path = "data/TEST/annotations/" + \
            (temp[len(temp) - 1]).split('.')[0] + ".xml"
        ground_truth = parseAnnotatedData(annotation_path)
        for g in range(len(ground_truth)):
            ground_truth[g]['filename'] = test_image_path
            annotation_data.append(ground_truth[g])

    with open('annotation.txt', 'w') as result_file:
        for truth in annotation_data:
            box = truth['box']
            file = truth['filename']

            line = str(file) + " " + str(box[0]) + " " + str(box[1]) + \
                " " + str(box[0] + box[2]) + " " + str(box[1] + box[3]) + "\n"

            result_file.write(line)

    return


def exportDetectionData(testImageFolderPath, label):
    baxterDetector = detector.BaxterDetector()
    # baxterDetector.threshold = 2
    test_image_path_list = glob.glob(testImageFolderPath + "*.jpg")
    data = []

    for i in range(len(test_image_path_list)):
        test_image_path = test_image_path_list[i]

        temp = test_image_path.split('/')
        annotation_path = "data/TEST/annotations/" + \
            (temp[len(temp) - 1]).split('.')[0] + ".xml"

        boundingBoxData = baxterDetector.detectObject(label, test_image_path)
        for i in range(len(boundingBoxData)):
            boundingBoxData[i]['filename'] = test_image_path
            data.append(boundingBoxData[i])

    with open('detection_result.txt', 'w') as result_file:
        for boxData in data:
            box = boxData['box']
            confidence = boxData['confidence']

            print("-----------------")
            print(boxData['confidence'])

            line = str(boxData['filename']) + " " + str(box[0]) + " " + str(box[1]) + \
                " " + str(box[0] + box[2]) + " " + str(box[1] + box[3]) + \
                " " + str(confidence) + "\n"

            result_file.write(line)

    return


def plotAverageDetectionPRCurve(annotationFolderPath, testImageFolderPath, label):
    baxterDetector = detector.BaxterDetector()
    threshold_list = [0.1, 0.5, 1, 1.5,  2,  2.5, 3, 3.5, 4]
    iou_list = [0.2, 0.3, 0.4, 0.5]
    test_image_path_list = glob.glob(testImageFolderPath + "*.jpg")
    annotation_path_list = glob.glob(annotationFolderPath + "*.xml")

    precision_data = np.zeros((len(test_image_path_list), 12))
    recall_data = np.zeros((len(test_image_path_list), 12))

    for i in range(len(test_image_path_list)):
        test_image_path = test_image_path_list[i]

        temp = test_image_path.split('/')
        annotation_path = "data/TEST/annotations/" + \
            (temp[len(temp) - 1]).split('.')[0] + ".xml"

        [recall_list, precision_list] = plotDetectionPRCurve(baxterDetector,
                                                             annotation_path, test_image_path, label)
        precision_data[i] = precision_list
        recall_data[i] = recall_list

    baxterDetector.terminateSession()

    mean_precision = np.mean(precision_data, axis=0)
    mean_recall = np.mean(recall_data, axis=0)

    mean_precision_list = [x for (y, x) in sorted(
        zip(mean_recall, mean_precision))]
    mean_recall.sort()

    mean_precision_list = np.insert(mean_precision_list, 0, 1)
    mean_recall = np.insert(mean_recall, 0, 0)

    mean_precision_list = np.insert(
        mean_precision_list, len(mean_precision_list), 0)
    mean_recall = np.insert(mean_recall, len(mean_recall), 1)

    print("======= AVERAGE PRECISION AND RECALL =============")
    print(mean_precision_list)
    print(mean_recall)

    plot_PR_curve("PR Curve",  mean_recall, mean_precision_list,
                  "recall", "precision", "precision-recall values")

    return


def plotDetectionPRCurve(baxterDetector, groundTruthPath, testImagePath, label):
    threshold_list = [0.1, 0.5, 1, 1.5,  2,  2.5, 3, 3.5, 4]
    precision_list = []
    recall_list = []

    for threshold in threshold_list:
        baxterDetector.threshold = threshold
        boundingBoxData = baxterDetector.detectObject(label, testImagePath)
        annotatedData = parseAnnotatedData(groundTruthPath)

        precision = calculatePrecision(label, boundingBoxData, annotatedData)
        recall = calculateRecall(label, boundingBoxData, annotatedData)

        print("precision : " + str(precision) + " /  recall : " + str(recall))
        if precision == 0 and recall == 0:
            print("note! both precision and recall is 0... on file : " +
                  str(testImagePath))

        recall_list.append(recall)
        precision_list.append(precision)

    # precision_list = [x for (y, x) in sorted(
    #     zip(recall_list, precision_list))]
    # recall_list.sort()

    # plot_PR_curve("PR Curve", recall_list, precision_list,
    #               "recall", "precision", "precision-recall values")

    return [recall_list, precision_list]


def plot_PR_curve(title, recall_x, precision_y, x_label, y_label, label, y_min=0, y_max=1):
    import matplotlib.pyplot as plt
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
    # baxterDetector = detector.BaxterDetector()
    # plotDetectionPRCurve(baxterDetector,
    #                      "data/TEST/annotations/frame0002.xml", "data/TEST/frame0002.jpg", 0)
    # plotAverageDetectionPRCurve("data/TEST/annotations/", "data/TEST/", 0)

    exportDetectionData("data/TEST/", 0)
    # exportAnnotationData("data/TEST/")
    # parseAnnotatedData("data/TEST/annotations/frame0002.xml")
