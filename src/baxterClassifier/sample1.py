"""
mnist_softmax.py

Introduction to Neural Networks and Backpropagation, for the MNIST Task.

Implement a Softmax Linear Classifier (the same model as is used in the TF Tutorial) from scratch,
using only Numpy.
"""
import numpy as np
import gzip

TRAIN_FILE = 'data/train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'data/train-labels-idx1-ubyte.gz'
TEST_FILE = 'data/t10k-images-idx3-ubyte.gz'
TEST_LABELS = 'data/t10k-labels-idx1-ubyte.gz'


def _read32(bytestrm):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestrm.read(4), dtype=dt)[0]

print 'Extracting', TRAIN_FILE
with open(TRAIN_FILE, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    _read32(bytestream)
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    train_images = np.frombuffer(buf, dtype=np.uint8) / 255.0
    train_images = train_images.reshape(num_images, 784)

print 'Extracting', TRAIN_LABELS
with open(TRAIN_LABELS, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    _read32(bytestream)
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)

print 'Extracting', TEST_FILE
with open(TEST_FILE, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    _read32(bytestream)
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    test_images = np.frombuffer(buf, dtype=np.uint8) / 255.0
    test_images = test_images.reshape(num_images, 784)

print 'Extracting', TEST_LABELS
with open(TEST_LABELS, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    _read32(bytestream)
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    test_labels = np.frombuffer(buf, dtype=np.uint8)

# Set up Parameters --> Model is calculated as follows: softmax(Wx + b),
# where W, b are model params
input_size, num_classes = 784, 10
W, b = np.zeros((input_size, num_classes)), np.zeros((1, num_classes))
batch_size, steps = 1, 10000
learning_rate = 0.5

for i in xrange(steps):
    # Get batch of training data for current train_step
    indices = np.random.choice(train_images.shape[0], batch_size)
    x, y = train_images[indices].astype(
        np.float32, copy=False), np.array(labels[indices])

    # Forward Propagation
    scores = np.dot(x, W) + b
    softmax = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    # Compute the Loss (Cross-Entropy)
    loss = np.sum(-np.log(softmax[range(batch_size), y])) / batch_size
    if i % 100 == 0:
        print "Iteration %s: Loss %s" % (str(i), str(loss))

    # Back Propagation (dL/dScores is just softmax - 1(y = k) -> softmax minus
    # label indicator)
    dScores = softmax
    dScores[range(batch_size), y] -= 1
    dScores /= batch_size

    # Parameter Gradients (W, b)
    dW = np.dot(x.T, dScores)
    dB = np.sum(dScores, axis=0, keepdims=True)

    # Update Parameters with Gradients * Learning Rate
    W += -learning_rate * dW
    b += -learning_rate * dB

# Evaluate Test Accuracy
scores = np.dot(test_images, W) + b
predicted_classes = np.argmax(scores, axis=1)
print "Test Accuracy: %s" % (str(np.mean(predicted_classes == test_labels)))
