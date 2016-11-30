from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from inception.dataset import Dataset
from six.moves import xrange  # pylint: disable=redefined-builtin
from PIL import Image
import os
import random
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import sys
import cv2
import numpy as np


IMAGE_SIZE = 224
CHANNELS = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
NUM_CLASSES = 2
BATCH_SIZE = 5

dataset_path = "/path/to/your/dataset/mnist/"
train_labels_file = "train-labels.csv"


def encode_label(label):
    return int(label)


def read_label_file(file):
    f = open(file, "r")
    filepaths = []
    labels = []
    for line in f:
        filepath, label = line.split(",")
        filepaths.append(filepath)
        labels.append(encode_label(label))
    return filepaths, labels


def getTrainingInput():
    # reading labels and file path
    train_filepaths, train_labels = read_label_file(
        dataset_path + train_labels_file)

    # transform relative path into full path
    train_filepaths = [dataset_path + fp for fp in train_filepaths]

    # for this example we will create or own test partition
    all_filepaths = train_filepaths
    all_labels = train_labels

    # convert string into tensors
    train_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
    train_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)

    # create input queues
    train_input_queue = tf.train.slice_input_producer(
        [train_images, train_labels])

    # process path and string tensor into an image and a label
    file_content = tf.read_file(train_input_queue[0])
    train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
    train_label = train_input_queue[1]

    # define tensor shape
    train_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, CHANNELS])

    # collect batches of images before processing
    train_image_batch, train_label_batch = tf.train.batch(
        [train_image, train_label], batch_size=BATCH_SIZE)

    return train_image_batch, train_label_batch


def inputs(eval_data, data_dir, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_data(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)


def getBatchInput():

    # Make a queue of file names including all the JPEG images files in the relative
    # image directory.
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(".data/images/*.jpg"))

    # Read an entire image file
    image_reader = tf.WholeFileReader()

    # Read a whole file from the queue, the first returned value in the tuple is the
    # filename which we are ignoring.
    _, image_file = image_reader.read(filename_queue)

    # Decode the image as a JPEG file, this will turn it into a Tensor which we can
    # then use in training.
    image = tf.image.decode_jpeg(image_file)

    # Generate batch
    num_preprocess_threads = 1
    min_queue_examples = 256
    batch_size = 32

    images_batch = tf.train.shuffle_batch(
        [image],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    return images_batch


def encodeImg(filename):
    # image = Image.open(filename)
    # image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
    # img = np.array(image)
    try:
        print("===========================")
        img = cv2.imread(filename.strip())
        print(img)

        if img is not None:
            img_resized = cv2.resize(
                img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        else:
            print("cannot read image file : ", filename)
            return None

    except Exception as e:
        print("=======       EXCEPTION     ======= : ", e)
        return None

    img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_resized_np = np.asarray(img_RGB)
    inputs = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype='float32')
    inputs[0] = (img_resized_np / 255.0) * 2.0 - 1.0
    return inputs


def pretrain_read_next(csvFileName, batchSize, batchIndex):
    ins = open(csvFileName)
    lines = ins.readlines()
    startIndex = batchIndex * batchSize
    endIndex = (batchIndex + 1) * batchSize
    if endIndex >= len(lines):
        endIndex = len(lines) - 1

    nextLines = lines[startIndex:endIndex]

    images = []
    annotations = []

    for line in nextLines:
        data = line.split(",")
        classLabel = data[0]
        ymin = data[1]
        ymax = data[2]
        xmin = data[3]
        xmax = data[4]
        img_filename = ("data/" + data[5]).strip()
        img = encodeImg(img_filename)

        if img is None:
            continue

        images.append(img)

        if classLabel == "n03384167":
            label = [0, 1]
        else:
            label = [1, 0]

        annotations.append(np.asarray(label))

    return [np.array(images), np.array(annotations)]


def read_next(csvFileName, batchSize, batchIndex):
    ins = open(csvFileName)
    lines = ins.readlines()
    startIndex = batchIndex * batchSize
    endIndex = (batchIndex + 1) * batchSize
    if endIndex >= len(lines):
        endIndex = len(lines) - 1

    nextLines = lines[startIndex:endIndex]

    images = []
    annotations = []

    for line in nextLines:
        data = line.split(",")
        classLabel = data[0]
        ymin = data[1]
        ymax = data[2]
        xmin = data[3]
        xmax = data[4]
        img_filename = ("data/" + data[5]).strip()
        img = encodeImg(img_filename)

        if img is None:
            continue

        images.append(img)
        print(classLabel)
        label = [classLabel, ymin, ymax, xmin, xmax]
        annotations.append(np.asarray(label))

        # file_contents = tf.read_file(img_filename)
        # img = tf.image.decode_png(file_contents)

    return [np.array(images), np.array(annotations)]


def read_my_file_format(filename_and_label_tensor):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.

    Returns:
      Two tensors: the decoded image, and the string label.
    """
    filename, label = tf.decode_csv(
        filename_and_label_tensor, [[""], [""]], " ")
    file_contents = tf.read_file(filename)
    img = tf.image.decode_png(file_contents)
    return img, label


def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)

    img, label = read_my_file_format(filename_queue)

    img_batch, label_batch = tf.train.shuffle_batch(
        [img, label], batch_size=batch_size)
    return img_batch, label_batch


def main(argvs):
    batch_size = 50
    read_next("data/csv", 1, 0)


if __name__ == '__main__':
    main(sys.argv)