import tensorflow as tf
import numpy as np
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from transform_nets import input_transform_net


def placeholder_inputs(batch_size, width, height):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, height, width, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value

    net = tf_util.conv2d(point_cloud, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn1', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, kernel_size=[3,3], stride=[1,1], scope='mp1')
    net1 = net

    net = tf_util.conv2d(point_cloud, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn2', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, kernel_size=[3, 3], stride=[1, 1], scope='mp2')
    net2 = net

    net = tf_util.conv2d(point_cloud, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, kernel_size=[3, 3], stride=[1, 1], scope='mp3')
    net3 = net

    net = tf_util.conv2d(point_cloud, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn4', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, kernel_size=[3, 3], stride=[1, 1], scope='mp4')
    net4 = net

    net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='agg', bn_decay=bn_decay)

    net = tf_util.max_pool2d(net, kernel_size=[3, 3], stride=[1, 1], scope='mp5')

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 10, activation_fn=None, scope='fc3')

    return net


def get_loss(pred, label):
    """ pred: B*NUM_CLASSES,
        label: B, """
    labels = tf.one_hot(indices=label, depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    classify_loss = tf.reduce_mean(loss)
    return classify_loss


if __name__ == '__main__':
    batch_size = 2
    width = 32
    height = 32
    channels = 3
    num_pt = 124
    pos_dim = 5

    input_feed = np.random.rand(batch_size, height, width, channels)
    label_feed = np.random.rand(batch_size)
    label_feed[label_feed >= 0.5] = 1
    label_feed[label_feed < 0.5] = 0
    label_feed = label_feed.astype(np.int32)

    # # np.save('./debug/input_feed.npy', input_feed)
    # input_feed = np.load('./debug/input_feed.npy')
    # print input_feed

    with tf.Graph().as_default():
        input_pl, label_pl = placeholder_inputs(batch_size, height, width, 3)
        pos = get_model(input_pl, tf.constant(True))
        # loss = get_loss(logits, label_pl, None)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {input_pl: input_feed, label_pl: label_feed}
            res1, res2 = sess.run([pos], feed_dict=feed_dict)
            print res1.shape
            print res1

            print res2.shape
            print res2












