from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the data set.
# TODO: change this
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


# Return input image and labels for training
def getInputData():
    return


def distorted_inputs():
    """Construct distorted input for CIFAR training using the Reader ops.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                    batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inputs(eval_data):
    """
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(eval_data=eval_data,
                                          data_dir=data_dir,
                                          batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(
            name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(images):
    """Build the YOLO model.
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      Logits to be calculated for loss function 
    """

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('cv1Weights',
                                             shape=[3, 3, 3, 16],
                                             stddev=5e-2,
                                             wd=0.0)

        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('cv2Weights',
                                             shape=[3, 3,  int(
                                                 pool1.get_shape()[3]), 32],
                                             stddev=5e-2,
                                             wd=0.0)

        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('cv3Weights',
                                             shape=[3, 3,  int(
                                                 pool2.get_shape()[3]), 64],
                                             stddev=5e-2,
                                             wd=0.0)

        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)

    # pool3
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool3')

    # conv4
    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('cv4Weights',
                                             shape=[3, 3,  int(
                                                 pool3.get_shape()[3]), 128],
                                             stddev=5e-2,
                                             wd=0.0)

        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu(
            'biases', [128], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)

    # pool4
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool4')

    # conv5
    with tf.variable_scope('conv5') as scope:
        kernel = _variable_with_weight_decay('cv5Weights',
                                             shape=[3, 3,  int(
                                                 pool4.get_shape()[3]), 256],
                                             stddev=5e-2,
                                             wd=0.0)

        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu(
            'biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope.name)

    # pool5
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool5')

    # conv6
    with tf.variable_scope('conv6') as scope:
        kernel = _variable_with_weight_decay('cv6Weights',
                                             shape=[3, 3,  int(
                                                 pool4.get_shape()[3]), 256],
                                             stddev=5e-2,
                                             wd=0.0)

        conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu(
            'biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv6 = tf.nn.relu(bias, name=scope.name)

    # pool6
    pool6 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool6')

    # conv7
    with tf.variable_scope('conv7') as scope:
        kernel = _variable_with_weight_decay('cv7Weights',
                                             shape=[3, 3,  int(
                                                 pool6.get_shape()[3]), 1024],
                                             stddev=5e-2,
                                             wd=0.0)

        conv = tf.nn.conv2d(pool6, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu(
            'biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv7 = tf.nn.relu(bias, name=scope.name)

    # conv8
    with tf.variable_scope('conv8') as scope:
        kernel = _variable_with_weight_decay('cv8Weights',
                                             shape=[3, 3,  int(
                                                 conv7.get_shape()[3]), 1024],
                                             stddev=5e-2,
                                             wd=0.0)

        conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu(
            'biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv8 = tf.nn.relu(bias, name=scope.name)

    # conv9
    with tf.variable_scope('conv9') as scope:
        kernel = _variable_with_weight_decay('cv9Weights',
                                             shape=[3, 3,  int(
                                                 conv8.get_shape()[3]), 1024],
                                             stddev=5e-2,
                                             wd=0.0)

        conv = tf.nn.conv2d(conv8, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu(
            'biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv9 = tf.nn.relu(bias, name=scope.name)

    # local1 (Fully connected layer)
    with tf.variable_scope('local1') as scope:
        # Move everything into depth so we can perform a single matrix
        # multiply.
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
            'biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) +
                            biases, name=scope.name)

    # local2 (Fully connected layer)
    with tf.variable_scope('local2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
            'biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) +
                            biases, name=scope.name)

    # local3 (Fully connected layer)
    with tf.variable_scope('local3') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
            'biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) +
                            biases, name=scope.name)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(
            tf.matmul(local4, weights), biases, name=scope.name)

    return softmax_linear


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss, global_step):
    """Train CIFAR-10 model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
