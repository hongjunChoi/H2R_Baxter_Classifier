import numpy as np
import tensorflow as tf
import cv2
import time
import sys
import inputProcessor
import baxterClassifier as baxter


class BaxterDetector:

    def __init__(self, argvs=[]):
        self.classifier = baxter.BaxterClassifier()
        self.training_file = "data/synthetic_train_data.csv"
        self.top_results = 3
        self.overlap_threshold = 0.7  # IOU threshold to combine two overlapping bounding box
        self.threshold = 0.7          # logit threshold to be considered as detection
        self.overlap_bonus = 1.1      # Coefficient applied when two boxes are merged
        [self.image_mean, self.image_std] = inputProcessor.getNormalizationData(
            self.training_file)

    def detectObject(self, label, imagePath):
        # Start Tensorflow Session
        predictions = []
        with self.classifier.sess as sess:

            self.classifier.saver = tf.train.Saver()
            print("weight file to restore ... : ",
                  self.classifier.weights_file)

            self.classifier.saver.restore(
                self.classifier.sess, self.classifier.weights_file)

            batch = inputProcessor.regionProposal(
                imagePath, self.classifier.img_size)

            if batch is None:
                print("wrong user input regarding image or labels ")
                raise ValueError(
                    'User input file path or lable contains error ')
                return

            original_img = batch[0]
            image_batch = batch[1]
            boundingBoxInfo = batch[2]
            batch_size = len(image_batch)

            # CREATE INPUT IMAGE BATCH
            input_image = np.zeros(
                [len(image_batch), self.classifier.img_size, self.classifier.img_size, 3])

            for x in range(batch_size):
                input_image[x] = (
                    image_batch[x] - self.image_mean) / self.image_std

            prediction = sess.run(self.classifier.logits, feed_dict={
                self.classifier.x: input_image,
                self.classifier.batch_size: batch_size,
                self.classifier.dropout_rate: 1})

            # filter correctly detected crops
            for y in range(batch_size):
                prob = prediction[y][label]
                if prob > self.threshold:
                    boundingBox = boundingBoxInfo[y]
                    box1 = [boundingBox[0], boundingBox[1], boundingBox[
                        0] + boundingBox[2], boundingBox[1] + boundingBox[3]]

                    # non maximal suppression , thresholding , and merging here
                    flag = True
                    for idx in range(len(predictions)):
                        p = predictions[idx]
                        box = p[1]
                        box_prob = p[0]
                        box2 = [box[0], box[1], box[
                            0] + box[2], box[1] + box[3]]
                        if self.overlap_threshold < inputProcessor.intersection_over_union(box1, box2):
                            flag = False
                            if box_prob > prob:
                                predictions[idx][0] = predictions[
                                    idx][0] * self.overlap_bonus

                            else:
                                predictions[idx] = [
                                    prob * self.overlap_bonus, boundingBox]
                            break
                    if flag:
                        predictions.append([prob, boundingBox])

            # sort crops by logit values
            predictions.sort(reverse=True)

            # visualize detection results
            # self.visualizeDetectionResult(original_img, predictions)

            boxData = []
            for box in predictions:
                boxData.append({'category': label, 'box': box[1]})

            return boxData

    def visualizeDetectionResult(self, image, boxData):
        for i in range(self.top_results):
            boundingBoxData = boxData[i]

            x = int(boundingBoxData[1][0])
            y = int(boundingBoxData[1][1])
            winW = int(boundingBoxData[1][2])
            winH = int(boundingBoxData[1][3])

            # if boundingBoxData[0] > threshold:
            cv2.rectangle(image, (x, y),
                          (x + winW, y + winH), (0, 255, 0), 5)

        cv2.imshow("Detection Results", image)
        cv2.waitKey(1000)
        time.sleep(1)

        return


def main(argvs):
    baxterDetector = BaxterDetector()
    while True:
        # GET USER INPUT
        try:
            img_filename = raw_input('image location: ')
            predictingClass = int(raw_input('class value: '))
        except:
            print("Incorrect user input..")
            continue

        try:
            baxterDetector.detectObject(predictingClass, img_filename)
        except ValueError as e:
            print("error in detection... error : ", e)
            continue


if __name__ == '__main__':
    main(sys.argv)
