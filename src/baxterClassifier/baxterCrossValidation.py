import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import time
import sys
import inputProcessor
import baxterClassifier as baxter


def main(argvs):
    [meanImage, std] = inputProcessor.getNormalizationData(
        "data/custom_train_data.csv")
    baxterClassifier = baxter.BaxterClassifier(argvs)
    top_results = 2  # number of crops to show for detection

    # Start Tensorflow Session
    with baxterClassifier.sess as sess:

        baxterClassifier.saver = tf.train.Saver()
        print("weight file to restore ... : ", baxterClassifier.weights_file)

        baxterClassifier.saver.restore(
            baxterClassifier.sess, baxterClassifier.weights_file)

        cv2.waitKey(1000)
        print("starting session... ")

        batch = inputProcessor.get_custom_dataset_batch(
            50, "data/custom_test_data.csv", meanImage, std)
        image_batch = batch[0]
        label_batch = batch[1]
        batch_size = len(label_batch)

        prediction = tf.argmax(baxterClassifier.logits, 1)
        trueLabel = np.argmax(label_batch, 1)

        result = sess.run(prediction, feed_dict={
            baxterClassifier.x: image_batch,
            baxterClassifier.batch_size: batch_size,
            baxterClassifier.dropout_rate: 1})

        print("=============")
        print(result)
        print(trueLabel)
        print("=============\n\n")

        train_accuracy = baxterClassifier.accuracy.eval(feed_dict={baxterClassifier.x: image_batch,
                                                                   baxterClassifier.y: label_batch,
                                                                   baxterClassifier.batch_size: batch_size,
                                                                   baxterClassifier.dropout_rate: 1})
        print("\nTest Accuracy %.2f \n\n" % (train_accuracy))

if __name__ == '__main__':
    main(sys.argv)
