import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import time
import sys
import inputProcessor


class BaxterClassifier:
    fromfile = None
    tofile_img = 'test/output.jpg'
    tofile_txt = 'test/output.txt'
    weights_file = 'tmp/modelfull.ckpt'
    imshow = True
    filewrite_img = False
    filewrite_txt = False
    threshold = 0.2
    iou_threshold = 0.5
    num_box = 2
    w_img = 640
    h_img = 480

    def __init__(self, argvs=[]):
        self.alpha = 0.1
        self.grid_size = 7
        self.num_labels = 2
        self.num_bounding_box = 2
        self.img_size = 224
        self.batch_size = 16
        self.uninitialized_var = []
        self.learning_rate = 1e-4

        self.sess = tf.Session()

        self.x = tf.placeholder(
            tf.float32, shape=[None, self.img_size * self.img_size])
        # Reshape Image to be of shape [batch, width, height, channel]
        self.x_image = tf.reshape(
            self.x, [-1, self.img_size, self.img_size, 3])

        self.y = tf.placeholder(tf.float32, shape=[None, self.num_labels])
        self.detection_y = tf.placeholder(
            tf.float32, shape=[None, 5])

        self.dropout_rate = tf.placeholder(tf.float32)

        # self.logits = self.build_pretrain_network()
        # self.loss_val = self.lossVal()
        # self.train_op = self.trainOps()

        self.detection_logits = self.build_networks()
        self.detection_loss_val = self.detection_loss()
        self.detection_train_op = self.detectionTrainOp()

        # Creat operations for computing the accuracy
        # self.correct_prediction = tf.equal(
        #     tf.argmax(self.detection_logits, 1), tf.argmax(self.y, 1))
        # self.accuracy = tf.reduce_mean(
        #     tf.cast(self.correct_prediction, tf.float32))

    def argv_parser(self, argvs):
        for i in range(1, len(argvs), 2):
            if argvs[i] == '-fromfile':
                self.fromfile = argvs[i + 1]
            if argvs[i] == '-tofile_img':
                self.tofile_img = argvs[i + 1]
                self.filewrite_img = True
            if argvs[i] == '-tofile_txt':
                self.tofile_txt = argvs[i + 1]
                self.filewrite_txt = True
            if argvs[i] == '-imshow':
                if argvs[i + 1] == '1':
                    self.imshow = True
                else:
                    self.imshow = False

    def build_pretrain_network(self):

        self.conv_1 = self.conv_layer(1, self.x_image, 32, 5, 1)
        self.pool_2 = self.pooling_layer(3, self.conv_1, 2, 2)
        self.conv_3 = self.conv_layer(2, self.pool_2, 32, 5, 1)
        self.pool_3 = self.pooling_layer(3, self.conv_3, 2, 2)

        self.conv_5 = self.conv_layer(4, self.pool_3, 64, 5, 1)
        self.conv_6 = self.conv_layer(5, self.conv_5, 64, 5, 1)
        self.pool_7 = self.pooling_layer(6, self.conv_6, 2, 2)

        self.fc_8 = self.fc_layer(
            25, self.pool_7, 1024, flat=True, linear=False)

        self.dropout_9 = self.dropout_layer(27, self.fc_8, self.dropout_rate)

        self.softmax_10 = self.softmax_layer(
            28, self.dropout_9, 1024, self.num_labels)

        return self.softmax_10

    def build_networks(self):

        self.conv_1 = self.conv_layer(1, self.x_image, 32, 5, 1)
        self.pool_2 = self.pooling_layer(3, self.conv_1, 2, 2)
        self.conv_3 = self.conv_layer(2, self.pool_2, 32, 5, 1)
        self.pool_3 = self.pooling_layer(3, self.conv_3, 2, 2)

        self.conv_5 = self.conv_layer(4, self.pool_3, 64, 5, 1)
        self.conv_6 = self.conv_layer(5, self.conv_5, 64, 5, 1)
        self.pool_7 = self.pooling_layer(6, self.conv_6, 2, 2)

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

        # Added detection network from below
        self.conv_25 = self.conv_layer(
            25, self.pool_7, 1024, 3, 1, initialize=True)
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
        # 7 * 7 * 12 =  588

        self.fc_32 = self.fc_layer(
            32, self.fc_30, 588, flat=False, linear=True, initialize=True)

        return self.fc_32

    def conv_layer(self, idx, inputs, filters, size, stride, initialize=False):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal(
            [size, size, int(channels), filters], stddev=0.1), name="weight" + str(idx))
        biases = tf.Variable(tf.constant(
            0.1, shape=[filters]), name="bias" + str(idx))

        conv = tf.nn.conv2d(inputs, weight, strides=[
                            1, stride, stride, 1], padding='SAME', name=str(idx) + '_conv')
        conv_biased = tf.add(conv, biases, name=str(idx) + '_conv_biased')

        if initialize:
            (self.uninitialized_var).append(weight)
            (self.uninitialized_var).append(biases)

        return tf.maximum(self.alpha * conv_biased, conv_biased, name=str(idx) + '_leaky_relu')

    def pooling_layer(self, idx, inputs, size, stride):
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME', name=str(idx) + '_pool')

    def dropout_layer(self, idx, inputs, dropout_rate):
        return tf.nn.dropout(inputs, dropout_rate)

    def fc_layer(self, idx, inputs, hiddens, flat=False, linear=False, initialize=False):
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
            0.1, shape=[hiddens]), name='fc_bias' + str(idx))

        if initialize:
            (self.uninitialized_var).append(weight)
            (self.uninitialized_var).append(biases)

        return tf.nn.relu(tf.add(tf.matmul(inputs_processed, weight), biases))

    def softmax_layer(self, idx, inputs, hidden, num_labels):
        weights = tf.Variable(tf.truncated_normal(
            [hidden, num_labels], stddev=1 / hidden))
        biases = tf.Variable(tf.constant(0.1, shape=[num_labels]))

        softmax_linear = tf.add(
            tf.matmul(inputs, weights), biases)
        return softmax_linear

    def detection_loss(self):
        outputData = (self.detection_logits).eval(
            session=self.sess)  # 7 * 7 * 12
        trueLabel = self.detection_y  # batch_size * 5 (class, 4 coordinates)

        # coord value
        yCoord = 5
        # noobj value
        yNoobj = .5

        loss_vector = []

        for index in range(self.batch_size):

            output = outputData[index]
            t = trueLabel[index]

            trueClass = t[0]
            if trueClass == "n04507155":
                trueClassIndex = 0
                true_class_probs = np.array([1, 0])

            elif trueClass == "n03642806":
                trueClassIndex = 1
                true_class_probs = np.array([0, 1])

            else:
                print("----- WRONG ----")
                trueClassIndex = 1
                true_class_probs = np.array([0, 1])

            # FIND THE CENTER POINT / WIDTH AND HEIGTH
            midX = ((t[3] + t[4]) / 2) / self.img_size
            midY = ((t[1] + t[2]) / 2) / self.img_size

            width = (t[4] - t[3]) / self.img_size
            height = (t[2] - t[1]) / self.img_size

            annotationBox = np.asarray([midX, midY, width, height])

            # PREDICTED Confidence score for each bounding box
            probs = np.zeros((7, 7, 2, 2))

            # class probabilities
            class_probs = np.zeros([7, 7, 2])
            class_probs = np.reshape(output, (7, 7, 2))
            scales = np.reshape(output[98:196], (7, 7, 2))

            # BOX center / location data
            boxes = np.reshape(output[196:], (7, 7, 2, 4))

            for i in range(2):
                for j in range(20):
                    probs[:, :, i, j] = np.multiply(
                        class_probs[:, :, j], scales[:, :, i])

            xval = 0
            wval = 0
            cval = 0
            noobjc = 0
            probc = 0

            for x in range(7):
                for y in range(7):
                    boxIndex = np.argmax(iou(boxes[x][y][0], annotationBox), iou(
                        boxes[x][y][1], annotationBox))

                    xdiff = 0
                    ydiff = 0
                    confidenceDiff = 0
                    reversedConfidencediff = 0

                    # predicted  box coordinate data of size 4
                    box = boxes[x][y][boxIndex]
                    predictedClass = np.argmax(class_probs[x][y])
                    predictedClassConfidence = class_probs[
                        x][y][predictedClass]

                    print("==== !!!! ======")
                    print(predictedClass)
                    print(trueClassIndex)

                    if predictedClass is trueClassIndex:
                        ###########################
                        ###########################
                        xdiff = midX - box[0]
                        # square the difference
                        xdiff = xdiff ** 2
                        ydiff = midY - box[1]
                        # square the difference
                        ydiff = ydiff ** 2
                        xval = xval + xdiff + ydiff

                        ###########################
                        ###########################
                        wdiff = math.sqrt(width) - \
                            math.sqrt(box[2])
                        # square the difference
                        wdiff = wdiff ** 2
                        hdiff = math.sqrt(height) - \
                            math.sqrt(box[3])
                        # square the difference
                        hdiff = hdiff ** 2
                        wval = wval + wdiff + hdiff

                        ###########################
                        ###########################
                        confidenceDiff = 0
                        trueConfidence = iou(annotationBox, box)
                        confidenceDiff = trueConfidence - predictedClassConfidence

                        # square the difference
                        confidenceDiff = confidenceDiff ** 2
                        cval += confidenceDiff

                    else:
                        trueConfidence = 0
                        reversedConfidencediff = trueConfidence - predictedClassConfidence

                        # square the difference
                        reversedConfidencediff = reversedConfidencediff ** 2
                        noobjc += reversedConfidencediff

                    # IF there is an object in x, y
                    gridCell = [(1 / 7) * x + 1 / 14, (1 / 7)
                                * y + 1 / 14, 1 / 7, 1 / 7]
                    if iou(annotationBox, np.array(gridCell)) > 0.5:
                        prob_difference_vector = (
                            class_probs[x][y] - true_class_probs)
                        probc = probc + \
                            np.sum(prob_difference_vector *
                                   prob_difference_vector)

            xval = xval * yCoord
            wval = wval * yCoord
            noobjc = noobjc * yNoobj
            loss_vector[index] = xval + wval + cval + noobjc + probc
        return tf.reduce_mean(loss_vector)

    def lossVal(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y))

    def trainOps(self):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val)

    def detectionTrainOp(self):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.detection_loss_val)

    def detect_from_cvmat(self, img):
        s = time.time()
        self.h_img, self.w_img, _ = img.shape
        img_resized = cv2.resize(img, (448, 448))
        img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray(img_RGB)
        inputs = np.zeros((1, 448, 448, 3), dtype='float32')
        inputs[0] = (img_resized_np / 255.0) * 2.0 - 1.0
        in_dict = {self.x: inputs}
        net_output = self.sess.run(self.fc_32, feed_dict=in_dict)
        self.result = self.interpret_output(net_output[0])
        self.show_results(img, self.result)
        strtime = str(time.time() - s)

    def detect_from_file(self, filename):
        img = cv2.imread(filename)
        self.detect_from_cvmat(img)

    def interpret_output(self, output):
        probs = np.zeros((7, 7, 2, 20))
        class_probs = np.reshape(output[0:980], (7, 7, 20))
        scales = np.reshape(output[980:1078], (7, 7, 2))
        boxes = np.reshape(output[1078:], (7, 7, 2, 4))

        offset = np.transpose(np.reshape(
            np.array([np.arange(7)] * 14), (2, 7, 7)), (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, 0:2] = boxes[:, :, :, 0:2] / 7.0
        boxes[:, :, :, 2] = np.multiply(boxes[:, :, :, 2], boxes[:, :, :, 2])
        boxes[:, :, :, 3] = np.multiply(boxes[:, :, :, 3], boxes[:, :, :, 3])

        boxes[:, :, :, 0] *= self.w_img
        boxes[:, :, :, 1] *= self.h_img
        boxes[:, :, :, 2] *= self.w_img
        boxes[:, :, :, 3] *= self.h_img

        for i in range(2):
            for j in range(20):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[
            0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[
                          i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

        return result

    def show_results(self, img, results):
        img_cp = img.copy()
        if self.filewrite_txt:
            ftxt = open(self.tofile_txt, 'w')
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3]) // 2
            h = int(results[i][4]) // 2

            if self.filewrite_img or self.imshow:
                cv2.rectangle(img_cp, (x - w, y - h),
                              (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(img_cp, (x - w, y - h - 20),
                              (x + w, y - h), (125, 125, 125), -1)
                cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][
                            5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if self.filewrite_txt:
                ftxt.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' +
                           str(w) + ',' + str(h) + ',' + str(results[i][5]) + '\n')

        if self.filewrite_img:
            cv2.imwrite(self.tofile_img, img_cp)

        if self.imshow:
            cv2.imshow('YOLO_small detection', img_cp)
            cv2.waitKey(1)

        if self.filewrite_txt:
            ftxt.close()

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)


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

        # Start Training Loop
        for i in range(85):
            print("starting  " + str(i) + "th  training iteration..")

            batch = inputProcessor.pretrain_read_next(
                "data/final_data.csv", batch_size, batch_index)

            batch_index = batch_index + 1

            image_batch = (batch[0][:, 0, :, :, :])
            label_batch = batch[1]

            # if i % 3 == 0:
            #     prediction = tf.argmax(baxterClassifier.logits, 1)
            #     print(sess.run(prediction, feed_dict={
            #           baxterClassifier.x_image: image_batch,
            #           baxterClassifier.dropout_rate: 1}))

            #     train_accuracy = baxterClassifier.accuracy.eval(feed_dict={baxterClassifier.x_image: image_batch,
            #                                                                baxterClassifier.detection_y: label_batch,
            # baxterClassifier.dropout_rate: 1})

            #     print("Step %d, Training Accuracy %.2f" % (i,
            #                                                train_accuracy))

            baxterClassifier.detection_train_op.run(feed_dict={baxterClassifier.x_image: image_batch,
                                                               baxterClassifier.detection_y: label_batch,
                                                               baxterClassifier.dropout_rate: 0.5})

        save_path = baxterClassifier.saver.save(sess, "tmp/modelfull.ckpt")
        print("saving model to ", save_path)


if __name__ == '__main__':
    main(sys.argv)
