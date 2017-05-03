import numpy as np
import tensorflow as tf
import cv2

CIFAR_IMG_SIZE = 32
IMAGE_SIZE = 64
CHANNELS = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
NUM_CLASSES = 2


def build_pretrain_network(x, dropout_rate, num_labels):

    conv_1 = conv_layer(1, x, 32, 3, 1)
    conv_2 = conv_layer(2, conv_1, 32, 3, 1)
    pool_3 = pooling_layer(3, conv_2, 2, 2)

    conv_4 = conv_layer(4, pool_3, 64, 3, 1)
    conv_5 = conv_layer(5, conv_4, 64, 3, 1)
    pool_6 = pooling_layer(6, conv_5, 2, 2)

    fc_25 = fc_layer(25, pool_6, 4096, flat=True)
    dropout_26 = dropout_layer(26, fc_25, dropout_rate)

    fc_27 = fc_layer(27, dropout_26, 4096, flat=False)
    dropout_28 = dropout_layer(28, fc_27, dropout_rate)

    fc_29 = fc_layer(29, dropout_28, 1024, flat=False)
    dropout_30 = dropout_layer(30, fc_29, dropout_rate)

    softmax_31 = softmax_layer(
        31, dropout_30, 1024, num_labels)

    return softmax_31


def conv_layer(varIndex, inputs, filters, size, stride, initialize=False):
    channels = inputs.get_shape()[3]
    weight = tf.Variable(tf.truncated_normal(
        [size, size, int(channels), filters], stddev=0.1), name="weight" + str(varIndex))

    biases = tf.Variable(tf.constant(
        0.1, shape=[filters]), name="bias" + str(varIndex))

    conv = tf.nn.conv2d(inputs, weight, strides=[
                        1, stride, stride, 1], padding='SAME', name=str(varIndex) + '_conv')
    conv_biased = tf.add(conv, biases, name=str(varIndex) + '_conv_biased')

    return tf.nn.relu(conv_biased)


def pooling_layer(varIndex, inputs, size, stride):
    return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME', name=str(varIndex) + '_pool')


def dropout_layer(varIndex, inputs, dropout_rate):
    return tf.nn.dropout(inputs, dropout_rate)


def fc_layer(varIndex, inputs, hiddens, flat=False, initialize=False):
    input_shape = inputs.get_shape().as_list()

    if flat:
        inputs_processed = tf.reshape(inputs, [32, -1])
        dim = input_shape[1] * input_shape[2] * input_shape[3]

    else:
        dim = input_shape[1]
        inputs_processed = inputs

    weight = tf.Variable(tf.truncated_normal(
        [dim, hiddens], stddev=0.1))

    biases = tf.Variable(tf.constant(
        0.1, shape=[hiddens]), name='fc_bias' + str(varIndex))

    return tf.nn.relu(tf.add(tf.matmul(inputs_processed, weight), biases))


def softmax_layer(varIndex, inputs, hidden, num_labels):
    weights = tf.Variable(tf.truncated_normal(
        [hidden, num_labels], stddev=1 / hidden))

    biases = tf.Variable(tf.constant(0.1, shape=[num_labels]))

    softmax_linear = tf.add(
        tf.matmul(inputs, weights), biases)
    return softmax_linear


def lossVal(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))


def trainOps(learning_rate, loss_val):
    return tf.train.AdamOptimizer(learning_rate).minimize(loss_val)


def getNormalizationData(trainingDataPath):

    with open(trainingDataPath, 'r') as f:
        num_lines = sum(1 for line in f)
        imageList = np.zeros([num_lines, IMAGE_SIZE, IMAGE_SIZE, 3])

    with open(trainingDataPath, 'r') as source:
        index = 0

        for line in source:
            path = line.split(",")[0]
            image = getImage(path)

            if image is None:
                continue

            img_resized = cv2.resize(
                image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

            imageList[index] = img_resized
            index = index + 1

        meanImage = np.mean(imageList, axis=0)
        std = np.std(imageList, axis=0)
        return [meanImage, std]


def get_custom_dataset_batch(batch_size, train_dataset_path, meanImage, std):
    image_batch = np.zeros([batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
    label_batch = np.zeros([batch_size, 2])
    with open(train_dataset_path, 'r') as source:
        data = [(np.random.rand(), line) for line in source]
        data.sort()

        count = 0
        index = 0

        while(count < batch_size):
            line = data[index][1].split(",")
            path = str(line[0])
            classLabel = int(line[1])
            image = getImage(path)

            if image is None:
                index = index + 1
                continue

            img_resized = cv2.resize(
                image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

            if classLabel is 0:
                label = [1, 0]
            else:
                label = [0, 1]

            processed_image = preprocessImage(img_resized, meanImage, std)

            image_batch[count] = processed_image
            label_batch[count] = label
            index = index + 1
            count = count + 1

    return [image_batch, label_batch]


def preprocessImage(image, meanImage, std):
    return (image - meanImage) / std


def getImage(filename):

    try:
        img = cv2.imread(filename.strip())

        if img is None:
            return None

    except Exception as e:
        print("=======   EXCEPTION  ======= : ", filename, e)
        return None

    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = np.asarray(img_RGB)

    return image


if __name__ == '__main__':
    print("starting main function...")
    [meanImage, std] = getNormalizationData(
        "data/synthetic_train_data.csv")
    print("000")
    sess = tf.Session()

    print("session started....")

    weights_file = 'model/synthetic_model.ckpt'
    num_labels = 2
    img_size = 64
    uninitialized_var = []
    learning_rate = 1e-4
    weight_vars = []

    print("11111")

    x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
    y = tf.placeholder(tf.float32, shape=[None, num_labels])
    dropout_rate = tf.placeholder(tf.float32)

    print("22222")

    logits = build_pretrain_network(x, dropout_rate, num_labels)
    loss_val = lossVal(logits, y)
    train_op = trainOps(learning_rate, loss_val)

    print("3333")

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    print("444444")

    # INITIALIZE VARIABLES
    sess.run(tf.initialize_all_variables())
    print("5555555")

    batch_index = 0
    i = 0
    while batch_index < 50000:

        print("starting  " + str(i) + "th  with batch index :  " +
              str(batch_index) + "  training iteration..")
        i += 1

        ###################################################
        # GET BATCH FOR CUSTOM DATASET AND (FOR CALTECH DATASET)
        batch = get_custom_dataset_batch(
            32, "data/synthetic_train_data.csv", meanImage, std)
        image_batch = batch[0]
        label_batch = batch[1]
        batch_index = batch_index + 64
        batch_size = len(label_batch)

        # ###################################################
        print("66666")

        # # # PERIODIC PRINT-OUT FOR CHECKING
        # if i % 20 == 0:
        #     prediction = tf.argmax(logits, 1)
        #     trueLabel = np.argmax(label_batch, 1)

        #     result = sess.run(prediction, feed_dict={
        #         x: image_batch,
        #         # batch_size: batch_size,
        #         dropout_rate: 1})

        #     print("=============")
        #     print(result)
        #     print(trueLabel)
        #     print("=============\n\n")

        #     train_accuracy = sess.run(accuracy, feed_dict={x: image_batch,
        #                                                    y: label_batch,
        #                                                    dropout_rate: 1})

        #     print("\nStep %d, Training Accuracy %.2f \n\n" % (i,
        #                                                       train_accuracy))

        print("77777")

        # ACTUAL TRAINING PROCESS
        sess.run(train_op, feed_dict={x: image_batch,
                                      y: label_batch,
                                      dropout_rate: 0.5})
        print("8888888")

    # # DONE.. SAVE MODEL
    # print("99999")
    # save_path = saver.save(sess, weights_file)
    # print("saving model to ", save_path)
