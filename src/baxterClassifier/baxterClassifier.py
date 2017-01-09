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
        self.img_size = 64
        self.batch_size = tf.placeholder(tf.int32)
        self.uninitialized_var = []
        self.learning_rate = 1e-4

        # FOR ADAPTIVE LEARNING RATE
        self.global_step = tf.Variable(0)
        # self.learning_rate = tf.train.exponential_decay(
        #     0.1,                 # Base learning rate.
        #     self.global_step,    # Current index into the dataset.
        #     100,                 # Decay step.
        #     0.95,                # Decay rate.
        #     staircase=True)

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
        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val, global_step=self.global_step)


def main(argvs):

    baxterClassifier = BaxterClassifier(argvs)

    # Start Tensorflow Session
    with baxterClassifier.sess as sess:

        baxterClassifier.saver = tf.train.Saver()

        cv2.waitKey(1000)
        print("starting session... ")

        # INITIALIZE VARIABLES
        sess.run(tf.initialize_all_variables())

        # START TRAINING
        batch_index = 0
        i = 0
        while batch_index < 50000:

            print("starting  " + str(i) + "th  with batch index :  " +
                  str(batch_index) + "  training iteration..")
            i += 1

            ###################################################
            # GET BATCH (FOR CIFAR DATA SET)
            # batch_size = 50
            # batch = inputProcessor.get_next_cifar(batch_size, batch_index)
            # image_batch = batch[0]
            # label_batch = batch[1]
            # batch_index = batch_index + batch[2]
            # batch_size = len(label_batch)

            ###################################################
            # GET BATCH (FOR IMAGENET DATASET)
            # batch = inputProcessor.get_imagenet_batch(
            #     "data/train_data.csv", 10)
            # image_batch = batch[0]
            # label_batch = batch[1]
            # batch_index = batch_index + 100
            # batch_size = len(label_batch)

            ###################################################
            # GET BATCH (FOR CALTECH DATASET)
            batch = inputProcessor.get_caltech_dataset_batch(8)
            image_batch = batch[0]
            label_batch = batch[1]
            batch_index = batch_index + 100
            batch_size = len(label_batch)

            ###################################################

            ###################################################
            # GET BATCH (FOR CUSTOM DATASET) => BAXTER CUSTOM READ IMAGES
            #
            #

            ###################################################

            # PERIODIC PRINT-OUT FOR CHECKING
            if i % 10 == 0:
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
                print("\nStep %d, Training Accuracy %.2f \n\n" % (i,
                                                                  train_accuracy))

            # ACTUAL TRAINING PROCESS
            baxterClassifier.train_op.run(feed_dict={baxterClassifier.x: image_batch,
                                                     baxterClassifier.y: label_batch,
                                                     baxterClassifier.batch_size: batch_size,
                                                     baxterClassifier.dropout_rate: 0.5})

        # DONE.. SAVE MODEL
        save_path = baxterClassifier.saver.save(
            sess, baxterClassifier.weights_file)
        print("saving model to ", save_path)


if __name__ == '__main__':
    main(sys.argv)
