import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import time
import sys
import inputProcessor


class BaxterClassifier:

    def __init__(self, argvs=[]):
        self.weights_file = 'tmp/modelnew.ckpt'
        self.num_labels = 2
        self.img_size = 128
        self.batch_size = 1
        self.uninitialized_var = []
        self.learning_rate = 1e-4

        self.sess = tf.Session()

        self.x = tf.placeholder(
            tf.float32, shape=[None, self.img_size, self.img_size, 3])

        self.y = tf.placeholder(tf.float32, shape=[None, self.num_labels])

        self.dropout_rate = tf.placeholder(tf.float32)

        self.logits = self.build_pretrain_network()
        self.loss_val = self.lossVal()
        self.train_op = self.trainOps()

        self.correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32))

    def build_pretrain_network(self):

        self.conv_1 = self.conv_layer(1, self.x, 32, 3, 1)
        self.conv_2 = self.conv_layer(3, self.conv_1, 32, 3, 1)
        self.pool_3 = self.pooling_layer(4, self.conv_2, 2, 2)

        self.conv_4 = self.conv_layer(5, self.pool_3, 64, 3, 1)
        self.conv_5 = self.conv_layer(6, self.conv_4, 64, 3, 1)
        self.pool_6 = self.pooling_layer(7, self.conv_5, 2, 2)

        # self.conv_7 = self.conv_layer(8, self.pool_6, 128, 3, 1)
        # self.conv_8 = self.conv_layer(9, self.conv_7, 128, 3, 1)
        # self.pool_9 = self.pooling_layer(10, self.conv_8, 2, 2)

        # self.conv_10 = self.conv_layer(11, self.pool_9, 256, 3, 1)
        # self.conv_11 = self.conv_layer(12, self.conv_10, 256, 3, 1)
        # self.conv_12 = self.conv_layer(13, self.conv_11, 256, 3, 1)
        # self.pool_13 = self.pooling_layer(14, self.conv_12, 2, 2)

        self.fc_25 = self.fc_layer(25, self.pool_6, 4096, flat=True)
        self.dropout_26 = self.dropout_layer(26, self.fc_25, self.dropout_rate)

        self.fc_27 = self.fc_layer(27, self.dropout_26, 4096, flat=False)
        self.dropout_28 = self.dropout_layer(28, self.fc_27, self.dropout_rate)

        self.fc_29 = self.fc_layer(29, self.dropout_28, 1024, flat=False)
        self.dropout_30 = self.dropout_layer(30, self.fc_29, self.dropout_rate)

        self.softmax_31 = self.softmax_layer(
            31, self.dropout_30, 1024, self.num_labels)

        return self.softmax_31

    def conv_layer(self, varIndex, inputs, filters, size, stride, initialize=False):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal(
            [size, size, int(channels), filters], stddev=0.1), name="weight" + str(varIndex))
        biases = tf.Variable(tf.constant(
            0.1, shape=[filters]), name="bias" + str(varIndex))

        conv = tf.nn.conv2d(inputs, weight, strides=[
                            1, stride, stride, 1], padding='SAME', name=str(varIndex) + '_conv')
        conv_biased = tf.add(conv, biases, name=str(varIndex) + '_conv_biased')

        if initialize:
            (self.uninitialized_var).append(weight)
            (self.uninitialized_var).append(biases)

        return tf.nn.relu(conv_biased)

    def pooling_layer(self, varIndex, inputs, size, stride):
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME', name=str(varIndex) + '_pool')

    def dropout_layer(self, varIndex, inputs, dropout_rate):
        return tf.nn.dropout(inputs, dropout_rate)

    def fc_layer(self, varIndex, inputs, hiddens, flat=False, initialize=False):
        input_shape = inputs.get_shape().as_list()

        if flat:
            inputs_processed = tf.reshape(inputs, [self.batch_size, -1])
            dim = input_shape[1] * input_shape[2] * input_shape[3]

        else:
            dim = input_shape[1]
            inputs_processed = inputs

        weight = tf.Variable(tf.truncated_normal(
            [dim, hiddens], stddev=0.1))

        biases = tf.Variable(tf.constant(
            0.1, shape=[hiddens]), name='fc_bias' + str(varIndex))

        if initialize:
            (self.uninitialized_var).append(weight)
            (self.uninitialized_var).append(biases)

        return tf.nn.relu(tf.add(tf.matmul(inputs_processed, weight), biases))

    def softmax_layer(self, varIndex, inputs, hidden, num_labels):
        weights = tf.Variable(tf.truncated_normal(
            [hidden, num_labels], stddev=1 / hidden))
        biases = tf.Variable(tf.constant(0.1, shape=[num_labels]))

        softmax_linear = tf.add(
            tf.matmul(inputs, weights), biases)
        return softmax_linear

    def lossVal(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y))

    def trainOps(self):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val)


def cropDisplayImage(img, boundingBox):
    crop_img = img[int(boundingBox[1]):int(boundingBox[2]), int(
        boundingBox[3]):int(boundingBox[4])]
    cv2.imshow("cam", crop_img)
    cv2.waitKey(1)
    time.sleep(3)


def main(argvs):
    # variable = raw_input('input something!: ')

    baxterClassifier = BaxterClassifier(argvs)

    batch_size = baxterClassifier.batch_size
    batch_index = 0

    threshold = 15
    # predictingClass = int(argvs[1])
    # img_filename = argvs[2]
    predictions = []

    # Start Tensorflow Session
    with baxterClassifier.sess as sess:

        baxterClassifier.saver = tf.train.Saver()
        baxterClassifier.saver.restore(
            baxterClassifier.sess, baxterClassifier.weights_file)
        cv2.waitKey(1000)
        print("starting session... ")

        while True:
            img_filename = raw_input('image location: ')
            predictingClass = int(raw_input('class value: '))

            batch = inputProcessor.read_next_image_versions(img_filename)

            original_img = batch[0]
            image_batch = batch[1]
            boundingBoxInfo = batch[2]

            maxClassProb = 0
            predClass = 0
            predAreaIndx = 0

            for j in range(len(image_batch)):
                image_version = image_batch[j]

                input_image = np.zeros(
                    [1, baxterClassifier.img_size, baxterClassifier.img_size, 3])
                input_image[0] = image_version

                prediction = sess.run(baxterClassifier.logits, feed_dict={
                    baxterClassifier.x: input_image,
                    baxterClassifier.dropout_rate: 1})

                prob = np.amax(prediction)
                predClass = np.argmax(prediction)

                if predClass == predictingClass:

                    boundingBox = boundingBoxInfo[j]
                    predictions.append([prob, boundingBox])


            predictions.sort(reverse=True)

            for i in range(2):
                boundingBoxData = predictions[i]
                print(boundingBoxData)

                x = boundingBoxData[1][0]
                y = boundingBoxData[1][1]
                winW = boundingBoxData[1][2]
                winH = boundingBoxData[1][3]

                # if boundingBoxData[0] > threshold:
                cv2.rectangle(original_img, (x, y),
                              (x + winW, y + winH), (0, 255, 0), 2)

            cv2.imshow("Window", original_img)
            cv2.waitKey(1)
            time.sleep(10)
        



if __name__ == '__main__':
    main(sys.argv)
