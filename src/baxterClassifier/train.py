from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.examples.tutorials.mnist import input_data
import os.path
import time
import inputProcessor
import numpy as np
import tensorflow as tf
import baxterClassifier as classifier


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_steps', 10,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels for CIFAR-10.
        images, labels = classifier.getInputData()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = classifier.inference(images)

        # Calculate loss.
        loss = classifier.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = classifier.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session()
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
    # batches = inputProcessor.getImageBatch()

    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Start Tensorflow Session
    with tf.Session() as sess:
        baxterCnn = classifier.BaxterClassifier()
        sess.run(tf.initialize_all_variables())

        # Start Training Loop
        for i in range(20):
            batch = mnist_data.train.next_batch(batch_size)
            baxterCnn.train_op.run(feed_dict={baxterCnn.x: batch[0], baxterCnn.y: batch[1],
                                              baxterCnn.keep_prob: 0.5})

        # # Evaluate Test Accuracy
        # print "Test Accuracy %g" % baxterCnn.accuracy.eval(feed_dict={
        #     baxterCnn.x: mnist_data.test.images, baxterCnn.y: mnist_data.test.labels,
        #     baxterCnn.keep_prob: 1.0})


if __name__ == '__main__':
    tf.app.run()