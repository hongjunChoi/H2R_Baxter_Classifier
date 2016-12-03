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
        self.img_size = 32
        self.batch_size = 50
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

        batch_index = 0
        i = 0
        while batch_index < 50000:
            if i < 600:
                break

            print("starting  " + str(i) + "th  training iteration..")
            print(batch_index)

            i += 1

            # batch = inputProcessor.pretrain_read_next(
            #     "data/final_data.csv", batch_size)

            filename = ""
            if batch_index < 10000:
                filaname = "data/cifar/data_batch_1"
            elif batch_index < 20000:
                filaname = "data/cifar/data_batch_2"
            elif batch_index < 30000:
                filaname = "data/cifar/data_batch_3"
            elif batch_index < 40000:
                filaname = "data/cifar/data_batch_4"
            else:
                filaname = "data/cifar/data_batch_5"

            index = batch_index % 10000

            batch = inputProcessor.get_next_cifar(
                filaname, batch_size, index)

            image_batch = batch[0]
            label_batch = batch[1]
            batch_index = batch_index + batch[2]

            if i % 50 == 0:

                prediction = tf.argmax(baxterClassifier.logits, 1)
                result = sess.run(prediction, feed_dict={
                    baxterClassifier.x: image_batch,
                    baxterClassifier.dropout_rate: 1})

                print("\n\n\n===== PREDICTION ======")
                print(result)
                print("\n\n\n====== TRUE LABEL ======")
                print(label_batch)
                print("\n\n")

                train_accuracy = baxterClassifier.accuracy.eval(feed_dict={baxterClassifier.x: image_batch,
                                                                           baxterClassifier.y: label_batch,
                                                                           baxterClassifier.dropout_rate: 1})
                print("\nStep %d, Training Accuracy %.2f \n\n" % (i,
                                                                  train_accuracy))

            baxterClassifier.train_op.run(feed_dict={baxterClassifier.x: image_batch,
                                                     baxterClassifier.y: label_batch,
                                                     baxterClassifier.dropout_rate: 0.5})

        save_path = baxterClassifier.saver.save(sess, "tmp/modelnew.ckpt")
        print("saving model to ", save_path)


if __name__ == '__main__':
    main(sys.argv)