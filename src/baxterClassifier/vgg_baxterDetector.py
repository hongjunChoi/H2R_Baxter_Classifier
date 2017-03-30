import numpy as np
import tensorflow as tf
import cv2
import time
import sys
import inputProcessor
import baxterClassifier as baxter
from imagenet_classes import class_names
import vgg16 as v


def main(argvs):

    # Start Tensorflow Session
    with tf.Session() as sess:
        top_results = 3  # number of crops to show for detection
        threshold = 0.007
        overlap_bonus = 1.1
        overlap_threshold = 0.7

        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg = v.vgg16(imgs, 'model/vgg16_weights.npz', sess)

        while True:
            # GET USER INPUT
            predictions = []
            try:
                img_filename = str(raw_input('image location: '))
                predictingClass = str(raw_input('class value: '))
                batch = inputProcessor.regionProposal(img_filename, 224)
            except:
                print("Incorrect user input..")
                continue

            if batch is None:
                print("wrong user input regarding image or labels ")
                continue

            original_img = batch[0]
            image_batch = batch[1]
            boundingBoxInfo = batch[2]
            batch_size = len(image_batch)

            # CREATE INPUT IMAGE BATCH
            input_image = np.zeros([len(image_batch), 224, 224, 3])

            for x in range(batch_size):
                input_image[x] = image_batch[x]

            prediction = sess.run(vgg.probs, feed_dict={vgg.imgs: input_image})

            # filter correctly detected crops
            for y in range(batch_size):
                prob = prediction[y]
                preds = (np.argsort(prob)[::-1])[0:10]

                # ONLY CONTINUE IF THE CLASS WE ARE LOOKING FOR IS WITHIN TOP 5
                # PREDICTIONS
                detected = False
                detection_probability = 0

                for p in preds:
                    if predictingClass in class_names[p]:
                        detected = True
                        detection_probability = prob[p]

                if not detected:
                    continue

                if detection_probability > threshold:
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
                        if overlap_threshold < inputProcessor.intersection_over_union(box1, box2):
                            flag = False
                            if box_prob > detection_probability:
                                predictions[idx][0] = predictions[
                                    idx][0] * overlap_bonus

                            else:
                                predictions[idx] = [
                                    detection_probability * overlap_bonus, boundingBox]
                            break
                    if flag:
                        predictions.append(
                            [detection_probability, boundingBox])

            # sort crops by logit values
            predictions.sort(reverse=True)

            if len(predictions) < top_results:
                top_results = len(predictions)

            for i in range(top_results):
                boundingBoxData = predictions[i]
                x = int(boundingBoxData[1][0])
                y = int(boundingBoxData[1][1])
                winW = int(boundingBoxData[1][2])
                winH = int(boundingBoxData[1][3])

                # if boundingBoxData[0] > threshold:
                cv2.rectangle(original_img, (x, y),
                              (x + winW, y + winH), (0, 255, 0), 5)

            cv2.imshow("Window", original_img)
            cv2.waitKey(1000)
            time.sleep(1)


if __name__ == '__main__':
    main(sys.argv)
