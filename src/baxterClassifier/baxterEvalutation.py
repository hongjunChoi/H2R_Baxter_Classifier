import os
import glob
import sys
import numpy as np
import cPickle
import inputProcessor
import xml.etree.ElementTree as ET
import copy


CIFAR_IMG_SIZE = 32
IMAGE_SIZE = 64
CHANNELS = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
NUM_CLASSES = 2


def calculatePrecision(detectedBoxList, annotatedData):
    # TODO: over all detected box in detectedBoxList, get me the ones that are actually correct
    # that matches box in annotatedData
    groundTruth = copy.deepcopy(annotatedData)
    detectedList = copy.deepcopy(detectedBoxList)

    total = len(detectedList)
    count = 0
    for d in detectedList:
        for g in groundTruth:
            box0 = g['box']
            class0 = g['category']

            box1 = d['box']
            class1 = d['category']

            if class0 == class1 or class0 is None:
                if inputProcessor.intersection_over_union(box0, box1) > 0.5:
                    count = count + 1
                    groundTruth.remove(g)

    return float(count) / float(total)


def calculateRecall(detectedBoxList, annotatedData):
    # TODO: over all ground truth annotated data, get me the ones that are
    # actually detected.
    groundTruth = copy.deepcopy(annotatedData)
    detectedList = copy.deepcopy(detectedBoxList)

    total = len(groundTruth)
    count = 0

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

    return float(count) / float(total)


def parseAnnotatedData(xmlFilePath):
    # returns box data consistent with detection along with the category
    tree = ET.parse(xmlFilePath)
    root = tree.getroot()
    boxData = []
    for boundingBoxObject in root.findall('object'):
        box = []
        category = boundingBoxObject.find('name').text
        polygon = boundingBoxObject.find('polygon')
        points = polygon.findall('pt')
        x = int(points[0].find('x').text)
        y = int(points[0].find('y').text)
        w = int(points[1].find('x').text) - int(points[0].find('x').text)
        h = int(points[3].find('y').text) - int(points[0].find('y').text)
        box = [x, y, w, h]
        boxData.append({'category': category,
                        'box': box})

    return box


def plotRegionPRCurve():
    return


def plotDetectionPRCurve():
    return

if __name__ == '__main__':
    parseAnnotatedData("data/synthetic_test_annotations/test1.xml")
