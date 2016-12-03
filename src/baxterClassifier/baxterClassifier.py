import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import time
import sys
import inputProcessor


class BaxterClassifier:

    def __init__(self, argvs=[]):
        self.weights_file = 'tmp/modelfull.ckpt'
        self.num_labels = 2
        self.img_size = 112
<<<<<<< HEAD
        self.batch_size = 20
=======
        self.batch_size = 50
>>>>>>> ff71743fe3af8f78e8536c19176bf9855541df2f
        self.uninitialized_var = []
        self.learning_rate = 1e-4

        self.sess = tf.Session()

        self.x = tf.placeholder(
            tf.float32, shape=[None, self.img_size, self.img_size, 1])

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

<<<<<<< HEAD
        self.conv_1 = self.conv_layer(1, self.x_image, 64, 7, 1)
        # self.pool_2 = self.pooling_layer(2, self.conv_1, 3, 1)
        self.conv_3 = self.conv_layer(3, self.conv_1, 64, 5, 1)
        # self.pool_4 = self.pooling_layer(4, self.conv_3, 5, 1)

        self.conv_5 = self.conv_layer(5, self.conv_3, 32, 7, 1)
        self.conv_6 = self.conv_layer(6, self.conv_5, 32, 3, 1)
        self.conv_7 = self.conv_layer(7, self.conv_6, 128, 3, 1)
        # self.conv_8 = self.conv_layer(8, self.conv_7, 128, 1, 1)
        self.pool_9 = self.pooling_layer(9, self.conv_7, 2, 2)
        self.conv_10 = self.conv_layer(10, self.pool_9, 256, 1, 1)
        self.conv_11 = self.conv_layer(11, self.conv_10, 512, 3, 1)
        # self.conv_12 = self.conv_layer(12, self.conv_11, 256, 1, 1)
        # self.conv_13 = self.conv_layer(13, self.conv_12, 512, 3, 1)
        # self.conv_14 = self.conv_layer(14, self.conv_13, 256, 1, 1)
        # self.conv_15 = self.conv_layer(15, self.conv_14, 512, 3, 1)
        # self.conv_16 = self.conv_layer(16, self.conv_15, 256, 1, 1)
        # self.conv_17 = self.conv_layer(17, self.conv_16, 512, 3, 1)
        # self.conv_18 = self.conv_layer(18, self.conv_17, 512, 1, 1)
        # self.conv_19 = self.conv_layer(19, self.conv_18, 1024, 3, 1)
        # self.pool_20 = self.pooling_layer(20, self.conv_19, 2, 2)
        # self.conv_21 = self.conv_layer(21, self.pool_20, 512, 1, 1)
        # self.conv_22 = self.conv_layer(22, self.conv_21, 1024, 3, 1)
        # self.conv_23 = self.conv_layer(23, self.conv_22, 512, 1, 1)
        # self.conv_24 = self.conv_layer(24, self.conv_23, 1024, 3, 1)

        self.fc_25 = self.fc_layer(
            25, self.conv_11, 512, flat=True, linear=False)
        self.fc_26 = self.fc_layer(
            25, self.fc_25, 256, flat=False, linear=False)

        self.softmax_26 = self.softmax_layer(
            26, self.fc_26, 256, self.num_labels)

        return self.softmax_26

    def build_networks(self):

        self.conv_1 = self.conv_layer(1, self.x_image, 64, 7, 2)
        self.pool_2 = self.pooling_layer(2, self.conv_1, 3, 2)
        self.conv_3 = self.conv_layer(3, self.pool_2, 64, 3, 1)
        self.pool_4 = self.pooling_layer(4, self.conv_3, 3, 2)
        self.conv_5 = self.conv_layer(5, self.pool_4, 128, 1, 1)
        self.conv_6 = self.conv_layer(6, self.conv_5, 256, 3, 1)
        self.conv_7 = self.conv_layer(7, self.conv_6, 256, 1, 1)
        self.conv_8 = self.conv_layer(8, self.conv_7, 512, 3, 1)
        self.pool_9 = self.pooling_layer(9, self.conv_8, 2, 2)
        self.conv_10 = self.conv_layer(10, self.pool_9, 256, 1, 1)
        self.conv_11 = self.conv_layer(11, self.conv_10, 512, 3, 1)
        self.conv_12 = self.conv_layer(12, self.conv_11, 256, 1, 1)
        self.conv_13 = self.conv_layer(13, self.conv_12, 512, 3, 1)
        self.conv_14 = self.conv_layer(14, self.conv_13, 256, 1, 1)
        self.conv_15 = self.conv_layer(15, self.conv_14, 512, 3, 1)
        self.conv_16 = self.conv_layer(16, self.conv_15, 256, 1, 1)
        self.conv_17 = self.conv_layer(17, self.conv_16, 512, 3, 1)
        self.conv_18 = self.conv_layer(18, self.conv_17, 512, 1, 1)
        self.conv_19 = self.conv_layer(19, self.conv_18, 1024, 3, 1)
        self.pool_20 = self.pooling_layer(20, self.conv_19, 2, 2)
        self.conv_21 = self.conv_layer(21, self.pool_20, 512, 1, 1)
        self.conv_22 = self.conv_layer(22, self.conv_21, 1024, 3, 1)
        self.conv_23 = self.conv_layer(23, self.conv_22, 512, 1, 1)
        self.conv_24 = self.conv_layer(24, self.conv_23, 1024, 3, 1)

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

        # Added detection network from below
        self.conv_25 = self.conv_layer(
            25, self.conv_24, 1024, 3, 1, initialize=True)
        self.conv_26 = self.conv_layer(
            26, self.conv_25, 1024, 3, 2, initialize=True)
        self.conv_27 = self.conv_layer(
            27, self.conv_26, 1024, 3, 1, initialize=True)
        self.conv_28 = self.conv_layer(
            28, self.conv_27, 1024, 3, 1, initialize=True)

        self.fc_29 = self.fc_layer(
            29, self.conv_28, 512, flat=True, linear=False, initialize=True)
        self.fc_30 = self.fc_layer(
            30, self.fc_29, 4096, flat=False, linear=False, initialize=True)

        # skip dropout_31
        # 7 * 7 * (2 * 5 + 20)
        # 7 * 7 * 12 = 7 * 7 * 588
        self.fc_32 = self.fc_layer(
            32, self.fc_30, 588, flat=False, linear=True, initialize=True)

        return self.fc_32

    def conv_layer(self, idx, inputs, filters, size, stride, initialize=False):
=======
        self.conv_1 = self.conv_layer(1, self.x, 32, 5, 1)
        self.conv_2 = self.conv_layer(3, self.conv_1, 32, 5, 1)
        self.pool_3 = self.pooling_layer(4, self.conv_2, 2, 2)

        self.conv_4 = self.conv_layer(6, self.pool_3, 64, 3, 1)
        self.conv_5 = self.conv_layer(7, self.conv_4, 64, 3, 1)
        self.pool_6 = self.pooling_layer(9, self.conv_5, 2, 2)

        self.fc_25 = self.fc_layer(25, self.pool_6, 2048, flat=True)
        self.fc_26 = self.fc_layer(26, self.fc_25, 1024, flat=False)

        self.dropout_27 = self.dropout_layer(27, self.fc_26, self.dropout_rate)
        self.softmax_28 = self.softmax_layer(
            28, self.dropout_27, 1024, self.num_labels)

        return self.softmax_28

    def conv_layer(self, varIndex, inputs, filters, size, stride, initialize=False):
>>>>>>> ff71743fe3af8f78e8536c19176bf9855541df2f
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


def main(argvs):

    baxterClassifier = BaxterClassifier(argvs)

    batch_size = baxterClassifier.batch_size
    batch_index = 0

    # Start Tensorflow Session
    with baxterClassifier.sess as sess:

        baxterClassifier.saver = tf.train.Saver()

        cv2.waitKey(1000)
        print("starting session... ")

        var = [v for v in tf.trainable_variables()]

        uninitialized_vars = []
        for var in tf.all_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        init_new_vars_op = tf.initialize_variables(uninitialized_vars)
        sess.run(init_new_vars_op)

        for i in range(100):
            print("starting  " + str(i) + "th  training iteration..")

            batch = inputProcessor.pretrain_read_next(
                "data/final_data1.csv", batch_size, batch_index)

            batch_index = batch_index + 1
            image_batch = batch[0]
            label_batch = batch[1]
            a = label_batch[:, 0].sum()
            b = len(label_batch[:, 0])-a
            countA += a
            countB += b
        
            if i > 90:
                prediction = tf.argmax(baxterClassifier.logits, 1)
                result = sess.run(prediction, feed_dict={
                    baxterClassifier.x: image_batch,
                    baxterClassifier.dropout_rate: 1})

                print("=================")
                print(result)
                print(label_batch)
                print("====================")

                train_accuracy = baxterClassifier.accuracy.eval(feed_dict={baxterClassifier.x: image_batch,
                                                                           baxterClassifier.y: label_batch,
                                                                           baxterClassifier.dropout_rate: 1})
                print("\nStep %d, Training Accuracy %.2f \n\n" % (i,
                                                                  train_accuracy))

            baxterClassifier.train_op.run(feed_dict={baxterClassifier.x: image_batch,
                                                     baxterClassifier.y: label_batch,
                                                     baxterClassifier.dropout_rate: 1})

        # save_path = baxterClassifier.saver.save(sess, "tmp/modelnew.ckpt")
        # print("saving model to ", save_path)

if __name__ == '__main__':
    main(sys.argv)
