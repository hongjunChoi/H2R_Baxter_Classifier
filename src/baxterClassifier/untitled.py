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
        self.batch_size = 50
        self.uninitialized_var = []
        self.learning_rate = 1e-4
        self.batch_size = 1

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

        self.logits = self.build_pretrain_network()
        self.loss_val = self.lossVal()
        self.train_op = self.trainOps()

        self.correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32))

        # self.detection_logits = self.build_networks()
        # self.detection_loss_val = self.detection_loss()
        # self.detection_train_op = self.detectionTrainOp()

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

        self.conv_1 = self.conv_layer(1, self.x_image, 64, 7, 2)
        self.pool_2 = self.pooling_layer(2, self.conv_1, 2, 2)
        self.conv_3 = self.conv_layer(3, self.pool_2, 192, 3, 1)
        self.pool_4 = self.pooling_layer(4, self.conv_3, 2, 2)
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

        hidden_dim = 512
        self.fc_25 = self.fc_layer(
            25, self.conv_24, hidden_dim, flat=True, linear=False)
        self.softmax_26 = self.softmax_layer(
            26, self.fc_25, hidden_dim, self.num_labels)

        return self.softmax_26

    def build_networks(self):

        self.conv_1 = self.conv_layer(1, self.x_image, 64, 7, 2)
        self.pool_2 = self.pooling_layer(2, self.conv_1, 2, 2)
        self.conv_3 = self.conv_layer(3, self.pool_2, 192, 3, 1)
        self.pool_4 = self.pooling_layer(4, self.conv_3, 2, 2)
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
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal(
            [size, size, int(channels), filters], stddev=0.1), name="weight" + str(idx))
        biases = tf.Variable(tf.constant(
            0.1, shape=[filters]), name="bias" + str(idx))

        pad_size = size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size],
                            [pad_size, pad_size], [0, 0]])
        inputs_pad = tf.pad(inputs, pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[
                            1, stride, stride, 1], padding='VALID', name=str(idx) + '_conv')
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
            dim = input_shape[1] * input_shape[2] * input_shape[3]
            inputs_transposed = tf.transpose(inputs, (0, 3, 1, 2))
            inputs_processed = tf.reshape(inputs_transposed, [-1, dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs
        weight = tf.Variable(tf.truncated_normal(
            [dim, hiddens], stddev=0.1), name='fc_weight' + str(idx))
        biases = tf.Variable(tf.constant(
            0.1, shape=[hiddens]), name='fc_bias' + str(idx))

        if initialize:
            (self.uninitialized_var).append(weight)
            (self.uninitialized_var).append(biases)

        if linear:
            return tf.add(tf.matmul(inputs_processed, weight), biases, name=str(idx) + '_fc')

        ip = tf.add(tf.matmul(inputs_processed, weight), biases)
        return tf.maximum(self.alpha * ip, ip, name=str(idx) + '_fc')

    def softmax_layer(self, idx, inputs, hidden, num_labels):
        weights = tf.Variable(tf.truncated_normal(
            [hidden, num_labels], stddev=1 / hidden))
        biases = tf.Variable(tf.constant(0.1, shape=[num_labels]))

        softmax_linear = tf.add(
            tf.matmul(inputs, weights), biases)
        return softmax_linear

     def detection_loss(self):

        # outputData = self.detection_logits  # 7 * 7 * 12
        # trueLabel = self.detection_y  # batch_size * 5 (class, 4 coordinates
        # )

        # output = outputData[0]
        # t = trueLabel[0]

        # # TODO: this assumes that there is only 1 box in annotation needs to be
        # # fixed

        # # FIND THE CENTER POINT / WIDTH AND HEIGTH
        # midX = (t[3] + t[4]) / 2
        # midY = (t[1] + t[2]) / 2

        # width = t[4] - t[3]
        # height = t[2] - t[1]

        # annotationBox = np.asarray([midX, midY, width, height])
        # # For each bounding box/ per grid -> check if object exists (in true
        # # label )

        # # Find "C" => IOU

        # probs = np.zeros((7, 7, 2, 2))

        # # class probabilities
        # class_probs = np.reshape(output, (7, 7, 2))
        # # C value is scales
        # scales = np.reshape(output[98:196], (7, 7, 2))
        # boxes = np.reshape(output[196:], (7, 7, 2, 4))

        # # coord value
        # yCoord = 5
        # # noobj value
        # yNoobj = .5
        # #self.y[x,y,w,h, C]
        # # find bounding box with higher IOU s
        # # need to do this still
        # box = 1

        # xval = 0
        # wval = 0
        # cval = 0
        # noobjc = 0
        # probc = 0
        # # jun check the synthax on the equations/how i'm getting values
        # # math should work but synthax isn't exact I don't think
        # # I also need help determing how I know if there is an object in
        # # the bounding box, right now I have a placeholder boolean value

        # for x in range(7):
        #     for y in range(7):
        #         for b in range(2):
        #             xdiff = 0
        #             ydiff = 0
        #             classDiff = 0
        #             reversedClassdiff = 0

        #             box = boxes[x][y][b]
        #             if iou(annotationBox, box) > 0.5:
        #                 ###########################
        #                 ###########################
        #                 xdiff = midX - boxes[i][j][box][0]
        #                 # square the difference
        #                 xdiff = xdiff ** 2
        #                 ydiff = midY - boxes[i][j][box][1]
        #                 # square the difference
        #                 ydiff = ydiff ** 2
        #                 xval = xval + xdiff + ydiff

        #                 ###########################
        #                 ###########################
        #                 wdiff = math.sqrt(width) - \
        #                     math.sqrt(boxes[i][j][box][2])
        #                 # square the difference
        #                 wdiff = wdiff ** 2
        #                 hdiff = math.sqrt(height) - \
        #                     math.sqrt(boxes[i][j][box][3])
        #                 # square the difference
        #                 hdiff = hdiff ** 2
        #                 wval = wval + wdiff + hdiff

        #                 ###########################
        #                 ###########################
        #                 classDiff = iou(annotationBox, boxes[i][j][
        #                     box]) - scales[i][j][box]

        #                 # square the difference
        #                 classDiff = classDiff ** 2
        #                 cval += classDiff

        #             else:
        #                 reversedClassdiff = iou(annotationBox, boxes[i][j][
        #                         box]) - scales[i][j][box]
        #                 # square the difference
        #                 reversedClassdiff = reversedClassdiff ** 2
        #                 noobjc += reversedClassdiff

        # xval = xval * yCoord
        # wval = wval * yCoord
        # noobjc = noobjc * yNoobj

        # return xval + wval + cval + noobjc + probc
        return tf.reduce_mean(self.detection_logits)

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

    # Read in data, write gzip files to "data/" directory
    mnist_data = input_data.read_data_sets("data/", one_hot=True)

    baxterClassifier = BaxterClassifier(argvs)

    batch_size = baxterClassifier.batch_size
    batch_index = 0

    # Start Tensorflow Session
    with baxterClassifier.sess as sess:

        baxterClassifier.saver = tf.train.Saver()

        # init_new_vars_op = tf.initialize_variables(
        #     baxterClassifier.uninitialized_var)
        # sess.run(init_new_vars_op)

        cv2.waitKey(1000)
        print("starting session... ")
        # sess.run(tf.initialize_all_variables())

        var = [v for v in tf.trainable_variables()]
        for i in range(len(var)):
            print(var[i].name)

        uninitialized_vars = []
        for var in tf.all_variables():
            try:
                sess.run(var)
                print("already init " + str(var.name))
            except tf.errors.FailedPreconditionError:
                print("initializing " + str(var.name))
                uninitialized_vars.append(var)

        # init_new_vars_op = tf.initialize_variables(uninitialized_vars)
        # sess.run(init_new_vars_op)
        # Start Training Loop
        for i in range(30):
            print("starting  " + str(i) + "th  training iteration..")
            batch2 = mnist_data.train.next_batch(batch_size)

            batch = inputProcessor.pretrain_read_next(
                "data/data.csv", batch_size, batch_index)

            batch_index = batch_index + 1
            
            print("------ batch ------")
            print(len(batch))
            print(len(batch2))
            print((batch[0][:, 0, :, :, :]).shape)
            print(batch[0].shape)
            print(batch2[0].shape)
            print(batch[1].shape)
            print(batch2[1].shape)

            if i % 5 == 0:
                train_accuracy = baxterClassifier.accuracy.eval(feed_dict={baxterClassifier.x_image: batch2[0],
                                                                           baxterClassifier.y: batch2[1],
                                                                           baxterClassifier.dropout_rate: 1.0})
                print("Step %d, Training Accuracy %.2f" % (i,
                                                           train_accuracy))
            baxterClassifier.train_op.run(feed_dict={baxterClassifier.x_image: batch2[0],
                                                     baxterClassifier.y: batch2[1],
                                                     baxterClassifier.dropout_rate: 0.5})

        save_path = baxterClassifier.saver.save(sess, "tmp/modelfull.ckpt")
        print("saving model to ", save_path)


if __name__ == '__main__':
    main(sys.argv)