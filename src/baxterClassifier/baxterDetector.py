import numpy as np
import tensorflow as tf
import cv2
import time
import sys
import inputProcessor
import baxterClassifier as baxter


def main(argvs):
    [meanImage, std] = inputProcessor.getNormalizationData(
        "data/custom_train_data.csv")
    baxterClassifier = baxter.BaxterClassifier(argvs)

    # Start Tensorflow Session
    with baxterClassifier.sess as sess:
        top_results = 2  # number of crops to show for detection
        baxterClassifier.saver = tf.train.Saver()
        print("weight file to restore ... : ", baxterClassifier.weights_file)

        baxterClassifier.saver.restore(
            baxterClassifier.sess, baxterClassifier.weights_file)

        cv2.waitKey(1000)
        print("starting session... ")

        while True:
            # GET USER INPUT
            predictions = []
            try:
                img_filename = raw_input('image location: ')
                predictingClass = int(raw_input('class value: '))
                batch = inputProcessor.regionProposal(img_filename)
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
            input_image = np.zeros(
                [len(image_batch), baxterClassifier.img_size, baxterClassifier.img_size, 3])

            for x in range(batch_size):
                input_image[x] = (image_batch[x] - meanImage) / std

            prediction = sess.run(baxterClassifier.logits, feed_dict={
                baxterClassifier.x: input_image,
                baxterClassifier.batch_size: batch_size,
                baxterClassifier.dropout_rate: 1})

            # filter correctly detected crops
            for y in range(batch_size):
                prob = prediction[y][predictingClass]
                boundingBox = boundingBoxInfo[y]
                predictions.append([prob, boundingBox])

            # sort crops by logit values
            predictions.sort(reverse=True)

            if len(predictions) < top_results:
                top_results = len(predictions)

            for i in range(top_results):
                boundingBoxData = predictions[i]
                print(boundingBoxData)

                x = int(boundingBoxData[1][0])
                y = int(boundingBoxData[1][1])
                winW = int(boundingBoxData[1][2])
                winH = int(boundingBoxData[1][3])

                # if boundingBoxData[0] > threshold:
                cv2.rectangle(original_img, (x, y),
                              (x + winW, y + winH), (0, 255, 0), 2)

            cv2.imshow("Window", original_img)
            cv2.waitKey(1000)
            time.sleep(1)


if __name__ == '__main__':
    main(sys.argv)
