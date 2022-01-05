
#--------------------CAE-DMD---------------

# import required packages
import numpy as np
import imageio
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf
import math as m
from numpy import linalg as LA
import h5py
import random
from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split
import glob
from PIL import Image
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import matplotlib.font_manager
from sklearn import svm
from pydmd import DMD
from sklearn.neighbors import KNeighborsClassifier

tf.compat.v1.disable_eager_execution()

# ---------------------------------
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# n_classes = 10
batch_size = 16
kp = tf.compat.v1.placeholder(tf.float32, shape=None)
# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.compat.v1.placeholder(tf.float32, [None, 64, 64, 3], name='InputData')  # place holder for test video

# x = tf.placeholder(tf.float32, [None, 784], name='InputData')  # place holder
# x = tf.ones([100, 784], dtype=tf.float32)
# 0-9 digits recognition => 10 classes
y = tf.compat.v1.placeholder(tf.float32, [None, 10], name='LabelData')
fc4d = tf.compat.v1.placeholder(tf.float32, [None, None], name='fc4d_v')
fc4d_cls = tf.compat.v1.placeholder(tf.float32, [None, None], name='fcd_cls')

enc1_d = tf.compat.v1.placeholder(tf.float32, [None, None], name='enc1_d')
enc2_d = tf.compat.v1.placeholder(tf.float32, [None, None], name='enc2_d')
DMD_modes = tf.compat.v1.placeholder(tf.float32, [None, None], name='dmd_modes')
# fcd4d1 = tf.placeholder(tf.float32, [None, None, None], name='fcd4d1')
is_training = tf.compat.v1.placeholder(tf.bool)
# print('input image = ' + str(x.shape))
# print('input image1 = ' + str(x[:1, :784].shape))


# ---- classify  -------
n_labels = 10
x_cls = tf.compat.v1.placeholder(tf.float32, [None, 64, 64, 3], name='InputData_cls')
Y_cls = tf.compat.v1.placeholder(tf.float32, [None, n_labels], name='Output_labels_cls')
# -----------------------

# This is
logs_path = "./logs/"
#   ---------------------------------
"""
We start by creating the layers with name scopes so that the graph in
the tensorboard looks meaningful
"""


#   ---------------------------------
def conv2d(input, name, kshape, strides=[1, 2, 2, 1]):
    with tf.compat.v1.name_scope(name):
        W = tf.compat.v1.get_variable(name='w_' + name,
                                      shape=kshape,
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                  mode="fan_avg",
                                                                                                  distribution="uniform"))
        b = tf.compat.v1.get_variable(name='b_' + name,
                                      shape=[kshape[3]],
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                  mode="fan_avg",
                                                                                                  distribution="uniform"))
        out = tf.nn.conv2d(input=input, filters=W, strides=strides, padding='SAME')
        out = tf.nn.bias_add(out, b)
        return out


def conv2dr(input, name, kshape, strides=[1, 1, 1, 1]):
    with tf.compat.v1.name_scope(name):
        W = tf.compat.v1.get_variable(name='w_' + name,
                                      shape=kshape,
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                  mode="fan_avg",
                                                                                                  distribution="uniform"))
        b = tf.compat.v1.get_variable(name='b_' + name,
                                      shape=[kshape[3]],
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                  mode="fan_avg",
                                                                                                  distribution="uniform"))
        out = tf.nn.conv2d(input=input, filters=W, strides=strides, padding='SAME')
        out = tf.nn.bias_add(out, b)
        out = tf.nn.leaky_relu(out)
        return out


# ---------------------------------
def deconv2d(input, name, kshape, n_outputs, strides=[1, 1]):
    with tf.compat.v1.name_scope(name):
        out = tf.contrib.layers.conv2d_transpose(input,
                                                 num_outputs=n_outputs,
                                                 kernel_size=kshape,
                                                 stride=strides,
                                                 padding='SAME',
                                                 weights_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                                     scale=1.0, mode="fan_avg", distribution="uniform", seed=0),
                                                 biases_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                                     scale=1.0, mode="fan_avg", distribution="uniform", seed=0),
                                                 activation_fn=tf.nn.relu)
        return out


#   ---------------------------------
def maxpool2d(x, name, kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    with tf.compat.v1.name_scope(name):
        out = tf.nn.max_pool2d(input=x,
                               ksize=kshape,  # size of window
                               strides=strides,
                               padding='SAME')
        return out


#   ---------------------------------
def upsample(input, name, factor=[2, 2]):
    size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
    with tf.compat.v1.name_scope(name):
        out = tf.image.resize(input, size=size, name=None, method=tf.image.ResizeMethod.BILINEAR)
        return out


#   ---------------------------------
def fullyConnected(input, name, output_size):
    with tf.compat.v1.name_scope(name):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size))
        W = tf.compat.v1.get_variable(name='w_' + name,
                                      shape=[input_size, output_size],
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                  mode="fan_avg",
                                                                                                  distribution="uniform"))
        b = tf.compat.v1.get_variable(name='b_' + name,
                                      shape=[output_size],
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                  mode="fan_avg",
                                                                                                  distribution="uniform"))
        input = tf.reshape(input, [-1, input_size])
        # out = tf.nn.relu(tf.add(tf.matmul(input, W), b))
        out = (tf.add(tf.matmul(input, W), b))
        print('\n\nWeights (W):\n\n', W)
        return out


def fullyConnectedf(input, name, output_size):
    with tf.compat.v1.name_scope(name):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size))
        W = tf.compat.v1.get_variable(name='w_' + name,
                                      shape=[input_size, output_size],
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                  mode="fan_avg",
                                                                                                  distribution="uniform"))
        b = tf.compat.v1.get_variable(name='b_' + name,
                                      shape=[output_size],
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                  mode="fan_avg",
                                                                                                  distribution="uniform"))
        input = tf.reshape(input, [-1, input_size])
        out = tf.nn.leaky_relu(tf.add(tf.matmul(input, W), b))
        # out = (tf.add(tf.matmul(input, W), b))
        print('\n\nWeights (W):\n\n', W)
        return out

    #   ---------------------------------


def dropout(input, name, keep_rate):
    with tf.compat.v1.name_scope(name):
        out = tf.nn.dropout(input, 1 - (keep_rate))
        return out


def my_pinv(a, rcond=1e-15):
    s, u, v = tf.linalg.svd(a)
    # Ignore singular values close to zero to prevent numerical overflow
    limit = rcond * tf.reduce_max(input_tensor=s)
    non_zero = tf.greater(s, limit)
    reciprocal = tf.compat.v1.where(non_zero, tf.math.reciprocal(s), tf.zeros(s.shape))
    lhs = tf.matmul(v, tf.linalg.diag(reciprocal))
    return tf.matmul(lhs, u, transpose_b=True)


#   ---------------------------------
# Let us now design the autoencoder

def fullyConnectedf_c(input, name, output_size):
    with tf.compat.v1.name_scope(name):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size))
        W = tf.compat.v1.get_variable(name='w_' + name,
                                      shape=[input_size, output_size],
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                  mode="fan_avg",
                                                                                                  distribution="uniform"))
        b = tf.compat.v1.get_variable(name='b_' + name,
                                      shape=[output_size],
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                  mode="fan_avg",
                                                                                                  distribution="uniform"))
        input = tf.reshape(input, [-1, input_size])
        out = tf.add(tf.matmul(input, W), b)
        # out = (tf.add(tf.matmul(input, W), b))
        print('\n\nWeights (W):\n\n', W)
        return out


def ConvAutoEncoderf(x, kp, name):
    with tf.compat.v1.name_scope(name):
        """
        We want to get dimensionality reduction of 784 to 196
        Layers:
            input --> 28, 28 (784)
            conv1 --> kernel size: (5,5), n_filters:25 ???make it small so that it runs fast
            pool1 --> 14, 14, 25
            dropout1 --> keeprate 0.8
            reshape --> 14*14*25
            FC1 --> 14*14*25, 14*14*5
            dropout2 --> keeprate 0.8
            FC2 --> 14*14*5, 196 --> output is the encoder vars
            FC3 --> 196, 14*14*5
            dropout3 --> keeprate 0.8
            FC4 --> 14*14*5,14*14*25
            dropout4 --> keeprate 0.8
            reshape --> 14, 14, 25
            deconv1 --> kernel size:(5,5,25), n_filters: 25
            upsample1 --> 28, 28, 25
            FullyConnected (outputlayer) -->  28* 28* 25, 28 * 28
            reshape --> 28*28
        """

        # input = tf.reshape(x, shape=[-1, 144, 176, 3])  # 240 x 320 downsampled
        input = tf.reshape(x, shape=[-1, 64, 64, 3])
        # assert isinstance(input.shape, object)
        print('input image1 = ' + str(input.shape))
        # fig = plt.figure()
        # plt.plot(input)
        # coding part

        # ================ Batch normalizatiomn ===========

        c1n = conv2dr(input, name='c1n', kshape=[5, 5, 3, 16])
        c11n = conv2dr(c1n, name='c11n', kshape=[5, 5, 16, 16])

        p1n = maxpool2d(c11n, name='p1n')

        c1 = conv2dr(p1n, name='c1', kshape=[5, 5, 16, 32])
        c11 = conv2dr(c1, name='c11', kshape=[5, 5, 32, 32])
        # c11 = tf.layers.batch_normalization(
        #     inputs=c11,
        #     axis=-1,
        #    momentum=0.999,
        #    epsilon=0.001,
        #     center=True,
        #     scale=True,
        #    training=is_training
        # )

        p1 = maxpool2d(c11, name='p1')
        # d1 = dropout(p1, name='drp1', keep_rate=0.5)
        # ================a

        # do1 = dropout(p1, name='do1', keep_rate=0.9)
        c21 = conv2dr(p1, name='c12', kshape=[5, 5, 32, 32])
        c22 = conv2dr(c21, name='c22', kshape=[5, 5, 32, 32])
        # p12 = maxpool2d(c22, name='p12')
        # d2 = dropout(p12, name='p12', keep_rate=0.5)
        # ================

        # do2 = dropout(p12, name='do2', keep_rate=0.9)
        c31 = conv2dr(c22, name='c31', kshape=[3, 3, 32, 64])
        c32 = conv2dr(c31, name='c32', kshape=[3, 3, 64, 64])
        p13 = maxpool2d(c32, name='p13')
        # d3 = dropout(p13, name='p12', keep_rate=0.5)
        # cmax1 = conv2dr(p13, name = 'cmax1', kshape = [5, 5, 128, 128])
        # ================
        c41 = conv2dr(p13, name='c41', kshape=[3, 3, 64, 64])
        c42 = conv2dr(c41, name='c42', kshape=[3, 3, 64, 128])
        # p14 = maxpool2d(c42, name='p14')

        c51 = conv2dr(c42, name='p41', kshape=[3, 3, 128, 128])
        c52 = conv2dr(c51, name='c52', kshape=[3, 3, 128, 128])
        do1 = dropout(c52, name='do1', keep_rate=kp)

        fc1 = fullyConnectedf(do1, name='fc1', output_size=8 * 8 * 128)

        return fc1

    # with tf.name_scope('cost'):
    #    cost1 = tf.reduce_mean(tf.square(tf.subtract(fc2, x)))


def my_dmdmn(fc2np, fc3np):
    # fc2np = np.transpose(fc2n)
    # fc3np = np.transpose(fc3n)
    nof = fc2np.shape
    nof = nof[1]

    X = fc2np
    Y = fc3np
    xc1 = fc2np
    xc2 = fc3np
    # xc1 = fc2np[:, 0: nof - 1]
    # xc2 = fc2np[:, 1: nof]

    [Uc, Sc, Vc] = np.linalg.svd(X, full_matrices=False)

    # print('Uc:', Uc)
    # print('Sc:', Sc)
    # print('Vc:', Vc)
    # Atil = dot(dot(dot(U.conj().T, Y), V), inv(Sig))
    Atilde = np.matmul(np.matmul(np.matmul(np.transpose(Uc), xc2), np.transpose(Vc)), np.linalg.pinv(np.diag(Sc)))
    # print('Atilde:', Atilde)

    [eigval1, W] = np.linalg.eig(Atilde)
    Phi1 = np.matmul(np.matmul(xc2, np.transpose(Vc)), np.linalg.pinv(np.diag(Sc)))
    Phi = np.matmul(Phi1, W)
    return eigval1, Phi, Atilde


def dmdmn(fc2n, fc3n):
    fc2np = fc2n
    fc3np = fc3n
    nof = fc2np.shape
    nof = nof[1]

    xc1 = fc2np
    xc2 = fc3np
    # xc1 = fc2np[:, 0: nof - 1]
    # xc2 = fc2np[:, 1: nof]

    [Uc, Sc, Vc] = np.linalg.svd(xc1, full_matrices=False)

    # print('Uc:', Uc)
    # print('Sc:', Sc)
    # print('Vc:', Vc)

    Atilde = np.matmul(np.matmul(np.matmul(np.transpose(Uc), xc2), np.transpose(Vc)), np.linalg.pinv(np.diag(Sc)))
    # print('Atilde:', Atilde)

    [eigval1, W] = np.linalg.eig(Atilde)
    # print('Eigval:', eigval1, '\n W:', W)

    Phi1 = np.matmul(np.matmul(xc2, np.transpose(Vc)), np.linalg.pinv(np.diag(Sc)))
    Phi = np.matmul(Phi1, W)
    # print('Phi:', Phi)

    t = np.linspace(0.0, 16 * m.pi, num=nof + 1)
    dt = t[2] - t[1]
    # print('t:', t)

    # [eigval1, W] = np.linalg.eig(Atilde)
    eigval = np.diag(np.log(eigval1))

    lambda1 = np.matrix.diagonal(eigval)
    omega = lambda1 / dt

    b = np.matmul(np.linalg.pinv(Phi), xc1[:, 0:1])
    # print('b:', b)

    X_bg = np.zeros((nof - 1, nof), dtype=np.float32)
    omegachk = np.zeros((omega.size, 1), dtype=complex)
    for k in range(2):
        omegachk[k] = omega[k]

    omega = omegachk
    # print('\n Omegachk:', omegachk)
    X_bg = np.zeros((omega.size, nof + 1), dtype=np.complex)

    for tt in range(nof + 1):
        # print('Running')
        omega_out = omegachk * t[tt]
        x_bgexp = np.exp(omega_out)
        X_bg[:, tt:tt + 1] = b * x_bgexp

    # print('x_bgexp:', x_bgexp)
    X_FG = np.matmul(Phi, X_bg)
    return np.real(X_FG)


def dmdm(fc2n, fc3n):
    # my_display(np.matmul(np.matmul(U, S),np.transpose(V)))
    # fc2n = np.array([[1, 2, 3, 1], [4, 5, 6, 2]], dtype=np.float32)
    fc2np = fc2n
    fc3np = fc3n
    nof = fc2np.shape
    nof = nof[1]

    xc1 = fc2np
    xc2 = fc3np
    # xc1 = fc2np[:, 0: nof - 1]
    # xc2 = fc2np[:, 1: nof]

    [Uc, Sc, Vc] = np.linalg.svd(xc1, full_matrices=False)

    # print('Uc:', Uc)
    # print('Sc:', Sc)
    # print('Vc:', Vc)

    Atilde = np.matmul(np.matmul(np.matmul(np.transpose(Uc), xc2), np.transpose(Vc)), np.linalg.pinv(np.diag(Sc)))
    # print('Atilde:', Atilde)

    [eigval1, W] = np.linalg.eig(Atilde)
    # print('Eigval:', eigval1, '\n W:', W)

    Phi1 = np.matmul(np.matmul(xc2, np.transpose(Vc)), np.linalg.pinv(np.diag(Sc)))
    Phi = np.matmul(Phi1, W)
    # print('Phi:', Phi)
    eigval = np.diag(np.log(eigval1))
    t = np.linspace(0.0, 16 * m.pi, num=nof)
    dt = t[2] - t[1]
    # print('t:', t)

    # [eigval1, W] = np.linalg.eig(Atilde)

    lambda1 = np.matrix.diagonal(eigval)
    omega = lambda1 / dt

    # print('\n \n Lambda:', lambda1)
    # print('Omega:', omega)
    # print('Omegasize:', omega.size)

    b = np.matmul(np.linalg.pinv(Phi), xc1[:, 0:1])
    # print('b:', b)

    X_bg = np.zeros((nof - 1, nof), dtype=np.float32)
    omegachk = np.zeros((omega.size, 1), dtype=complex)
    for k in range(2):
        omegachk[k] = omega[k]

    omega = omegachk
    # print('\n Omegachk:', omegachk)
    X_bg = np.zeros((omega.size, nof), dtype=np.complex)

    for tt in range(nof):
        # print('Running')
        omega_out = omegachk * t[tt]
        x_bgexp = np.exp(omega_out)
        X_bg[:, tt:tt + 1] = b * x_bgexp

    # print('x_bgexp:', x_bgexp)
    X_FG = np.matmul(Phi, X_bg)
    # print(np.real(X_FG))
    return np.real(X_FG)


def dmd(fc2np):
    fc2_x = np.transpose(fc2np)
    x1 = fc2_x[:, 0: 99]
    x2 = fc2_x[:, 1: 100]
    nof = np.shape(fc2_x)
    nof = nof[1]

    size_sig = np.shape(x1)
    [U, S, V] = np.linalg.svd(x1, full_matrices=False)
    # Atilde = tf.transpose(U)*x2*V/S
    # print(np.linalg.pinv(np.diag(S)))
    Atilde = np.matmul(np.matmul(np.matmul(np.transpose(U), x2), V), np.linalg.pinv(np.diag(S)))

    [eigval1, W] = np.linalg.eig(Atilde)
    eigval = np.diag(np.log(eigval1))
    # [W, Eigval] = tf.self_adjoint_eigval(Atilde)
    Phi1 = np.matmul(np.matmul(x2, V), np.linalg.pinv(np.diag(S)))
    Phi = np.matmul(Phi1, W)
    # t_interval = tf.linspace(0.0, 16*m.pi, 100, name="linspace")
    t = np.linspace(0.0, 16 * m.pi, num=nof)
    dt = t[2] - t[1]
    lambda1 = np.matrix.diagonal(eigval)
    omega = lambda1 / dt
    b = np.matmul(np.linalg.pinv(Phi), x1[:, 0:1])
    size_t = np.size(x1, 0)

    X_bg = np.zeros((nof - 1, nof), dtype=np.float32)
    omegachk = np.zeros((99, 1), dtype=complex)
    for k in range(99):
        omegachk[k] = omega[k]

    omega = omegachk

    for tt in range(nof):
        # print('Running')
        omega_out = omega * t[tt]
        x_bgexp = np.exp(omega_out)
        # nn = np.dot(b, x_bgexp)
        X_bg[:, tt:tt + 1] = b * x_bgexp
        # for j in range(1):
        #   print('2nd loop')
        #   chk = b * x_bgexp[:, j:j + 1]

    X_FG = np.matmul(Phi, X_bg)
    return np.real(X_FG)
    # return fc2np


def DeconAutoEncoder(fc2, name):
    fc3 = fullyConnected(fc2, name='fc3', output_size=14 * 14 * 5)
    do3 = dropout(fc3, name='do3', keep_rate=0.8)
    fc4 = fullyConnected(do3, name='fc4', output_size=14 * 14 * 25)
    do4 = dropout(fc4, name='do3', keep_rate=0.8)
    do4 = tf.reshape(do4, shape=[-1, 14, 14, 25])
    dc1 = deconv2d(do4, name='dc1', kshape=[5, 5], n_outputs=25)
    up1 = upsample(dc1, name='up1', factor=[2, 2])
    output = fullyConnected(up1, name='output', output_size=28 * 28)
    # rec = tf.nn.sigmoid(output)
    return output


def DeconAutoEncoderf(fc4de, kp, name):
    with tf.compat.v1.name_scope(name):
        #   ffdec1b = fullyConnected(fc4de, name='ffdec1b', output_size=72 * 96 * 3)
        #   do1dc = dropout(ffdec1b, name='do1dc', keep_rate=kp)

        #   ffdec1b1 = fullyConnected(do1dc, name='ffdec1b1', output_size=9 * 12 * 80)
        #   do1dc1 = dropout(ffdec1b1, name='do1dc1', keep_rate=kp)

        #   ffdec1b2 = fullyConnected(do1dc1, name='ffdec1b2', output_size=5 * 6 * 300)
        #   do1dc2 = dropout(ffdec1b2, name='do1dc3', keep_rate=kp)

        #   ffdec1b3 = fullyConnected(do1dc2, name='ffdec1b3', output_size=9 * 12 * 200)
        #   do1dc4 = dropout(ffdec1b3, name='do1dc4', keep_rate=kp)
        # ffdec1b2 = fullyConnected(ffdec1b1, name='ffdec1b2', output_size=8 * 8 * 2)
        # ffdec1b3 = fullyConnected(ffdec1b2, name='ffdec1b3', output_size=8 * 8 * 2)
        # ffdec1b4 = fullyConnected(ffdec1b3, name='ffdec1b4', output_size=8 * 8 * 2)
        # doc11 = dropout(ffdec1b, name='doc11', keep_rate=1)
        # ffdec2b = fullyConnected(ffdec1b, name='ffdec2b', output_size=15 * 20 * 15)
        # doc12 = dropout(ffdec2b, name='doc12', keep_rate=0.5)

        #  do4 = tf.reshape(do1dc1, shape=[-1, 36, 48, 1])
        # up1 = upsample(do4, name='up1', factor=[2, 2])
        # cd1 = conv2d(up1, name='c216d', kshape=[5, 5, 1000, 80])

        #   cd2d = conv2dr(do4, name='cd2d', kshape=[5, 5, 80, 40])
        #   cd2d2 = conv2dr(cd2d, name='cd2d2', kshape=[5, 5, 40, 40])
        # cd2d = deconv2d(do4, name='cd2d', kshape=[5, 5], n_outputs=3)
        #   up2d = upsample(cd2d2, name='up2d', factor=[2, 2])
        # cd2de = deconv2d(do4, name='cd2', kshape=[5, 5], n_outputs=400)
        # fdec2br = fullyConnected(cd2d, name='ffdec2br', output_size=15 * 20 * 40)
        # cd2dr = tf.reshape(fdec2br, shape=[-1, 15, 20, 40])

        #  cd3d = conv2dr(up2d, name='cd3d', kshape=[5, 5, 40, 40])
        #  cd3d2 = conv2dr(cd3d, name='cd3d2', kshape=[5, 5, 40, 20])
        # cd3d = deconv2d(up2d, name='cd3d', kshape=[5, 5], n_outputs=3)
        #  up3d = upsample(cd3d2, name='up3d', factor=[2, 2])

        # cd3d = deconv2d(up3d, name='c216d3', kshape=[5, 5], n_outputs=100)
        #  up4 = upsample(c216d3, name='up12', factor=[2, 2])

        #   c216d4 = conv2dr(up4, name='c216d4', kshape=[7, 7, 10, 10])
        #    c216d5 = conv2dr(c216d4, name='c216d5', kshape=[7, 7, 10, 3])

        #   up3 = upsample(c216d5, name='up3', factor=[2, 2])
        #   cd3d1 = conv2dr(up3, name='cd3d1', kshape=[5, 5, 20, 10])
        #   cd3d2f = conv2dr(cd3d1, name='cd3d2f', kshape=[5, 5, 10, 3])
        # cd3d = deconv2d(up4, name='cd3d', kshape=[5, 5], n_outputs=3)
        #   up35 = upsample(cd3d1, name='up35', factor=[2, 2])
        #   cd6 = conv2d(up35, name='c216d6', kshape=[5, 5, 16, 8])

        # up4 = upsample(cd6, name='up6', factor=[2, 2])
        #   cd7 = conv2d(cd6, name='c216d7', kshape=[5, 5, 8, 3])
        # cd8 = conv2d(cd7, name='c216d8', kshape=[5, 5, 1, 1])

        # cd1 = deconv2d(up1, name='cd1', kshape=[5, 5], n20outputs=60)
        # cd2 = deconv2d(up1, name='cd2', kshape=[5, 5], n_outputs=40)

        # up2 = upsample(cd1, name='up2', factor=[2, 2])
        # cd3 = conv2d(up2, name='c216d3', kshape=[5, 5, 50, 125])

        # up3 = upsample(cd3, name='up3', factor=[2, 2])
        # cd4 = conv2d(up3, name='c216d4', kshape=[5, 5, 30, 10])

        # up4 = upsample(cd4, name='up4', factor=[2, 2])
        # cd5 = deconv2d(up4, name='cd5', kshape=[5, 5], n_outputs=5)
        # cd41 = deconv2d(cd3, name='cd41', kshape=[5, 5], n_outputs=30)

        #     up3 = upsample(cd3, name='up3', factor=[2, 2])
        #     cd4 = deconv2d(up3, name='cd4', kshape=[5, 5], n_outputs=10)
        # cd5 = deconv2d(cd4, name='cd5', kshape=[5, 5], n_outputs=20)
        #     fdec1 = fullyConnected(fc4de, name='fdec1', output_size= 9 * 12 * 156)
        # fdec1 = fullyConnectedf(fc4de, name='fdec1', output_size= 8 * 10 * 1024)
        # do1dc1 = dropout(fdec1, name='do1dc1', keep_rate=kp)

        fdec2 = fullyConnectedf(fc4de, name='fdec2', output_size=8 * 8 * 128)
        # do1dc2 = dropout(fdec2, name='do1dc', keep_rate=kp)

        fd1res = tf.reshape(fdec2, shape=[-1, 8, 8, 128])

        # upd2 = upsample(fd1res, name='upd2', factor=[2, 2])
        cdec11 = conv2dr(fd1res, name='cdec11', kshape=[3, 3, 128, 128])
        cdec22 = conv2dr(cdec11, name='cdec22', kshape=[3, 3, 128, 128])

        upd3 = upsample(cdec22, name='upd3', factor=[2, 2])
        cdec1 = conv2dr(upd3, name='cdec1', kshape=[3, 3, 128, 64])
        cdec2 = conv2dr(cdec1, name='cdec2', kshape=[3, 3, 64, 64])

        # upd4 = upsample(cdec2, name='upd4', factor=[2, 2])
        # cdec3 = conv2dr(upd4, name='cdec3', kshape=[3, 3, 64, 64])
        # cdec4 = conv2dr(cdec3, name='cdec4', kshape=[3, 3, 64, 64])

        up4 = upsample(cdec2, name='up4', factor=[2, 2])
        cd4 = conv2dr(up4, name='c216d4', kshape=[5, 5, 64, 64])
        ffdec1b = conv2dr(cd4, name='c216d41', kshape=[5, 5, 64, 64])

        up5 = upsample(ffdec1b, name='up5', factor=[2, 2])

        cd5 = conv2dr(up5, name='cd5', kshape=[5, 5, 64, 64])
        cd6 = conv2dr(cd5, name='cd6', kshape=[5, 5, 64, 32])

        #     cd6 = deconv2d(up4, name='cd6', kshape=[5, 5], n_outputs=10)
        #      cd7 = deconv2d(cd6, name='cd7', kshape=[5, 5], n_outputs=10)
        # up6 = upsample(cd6, name='up6', factor=[2, 2])
        cd7 = conv2dr(cd6, name='cd7', kshape=[5, 5, 32, 32])
        cd8 = conv2dr(cd7, name='cd8', kshape=[5, 5, 32, 3])

        # ffdeco = fullyConnected(cd3d2f, name='ffdeco', output_size=192 * 256 * 3)
        # doout = tf.reshape(cd6, shape=[-1, 120, 160, 1])
        doout1 = tf.reshape(cd8, shape=[-1, 64, 64, 3])

        decoded = tf.sigmoid(doout1, name='recon')
        return doout1


# def DeconAutoEncoderco(x, fc2, name):
#   with tf.name_scope(name):
# with tf.Session() as sess1:
#   fcnum = sess1.run(fc2)

#      fc3 = fullyConnected(fc2, name='fc3', output_size=25 * 25 * 5)
#     do3 = dropout(fc3, name='do3', keep_rate=0.8)
#     fc4 = fullyConnected(do3, name='fc4', output_size=25 * 25 * 25)
#     do4 = dropout(fc4, name='do3', keep_rate=0.8)
#     do4 = tf.reshape(do4, shape=[-1, 25, 25, 25])
#     dc1 = deconv2d(do4, name='dc1', kshape=[5, 5], n_outputs=25)
#     up1 = upsample(dc1, name='up1', factor=[2, 2])
#     output = fullyConnected(up1, name='output', output_size=50 * 50)
# rec = tf.nn.sigmoid(output)
#     with tf.name_scope('cost'):
#         cost = tf.reduce_mean(tf.square(tf.subtract(output, x)))
#     return output, cost
# return output, cost, fc2, Atilde, Phi, S, U, V, W, eigval, x1, x2, lambda1, t_interval, omega


# ---------------Display Image-------------
def my_display(img):
    fig, axs = plt.subplots()
    axs.imshow(img)
    # axs[0, 1].imshow(img1
    # plt.subplot_tool()
    plt.show()


def my_imwrite(path, name, img):
    # imr = np.transpose(np.reshape(batch_test1[0:(m * n), 0:1], (n, m, 1)))
    # img = np.transpose(np.reshape(batch_test1[0:(m * n), 1:2], (n, m, 1)))
    # imb = np.transpose(np.reshape(batch_test1[0:(m * n), 2:3], (n, m, 1)))
    # imr = np.reshape(img[:, :, 0],[m,n])
    # imwr = np.zeros([m, n, 3])
    # imwr[:, :, 0] = np.reshape(img[:, :, 0],[m,n])
    # imwr[:, :, 1] = np.reshape(img[:, :, 1],[m,n])
    # imwr[:, :, 2] = np.reshape(img[:, :, 0],[m,n])
    # axs.imshow(imwr)
    # plt.imshow(imwr)
    imwr = img
    imwr1 = imwr.astype(np.float64) / imwr.max()  # normalize the data to 0 - 1
    imwr2 = 255 * imwr1  # Now scale by 255
    img = imwr2.astype(np.uint8)
    imageio.imwrite(path + name, img.astype(np.uint8))
    # imageio.imwrite(path + name + type, img.astype(np.uint8))


def my_imwrite_new(m, n, filenumber, c, batch_test1):
    # imr = batch_test1[:,:,0]
    # img = np.transpose(np.reshape(batch_test1[0:(m * n), 1:2], (n, m, 1)))
    # imb = np.transpose(np.reshape(batch_test1[0:(m * n), 2:3], (n, m, 1)))
    imwr = np.zeros([np.shape(batch_test1)[0], np.shape(batch_test1)[1], 3])
    imwr[:, :, 0] = batch_test1[:, :, 0]
    imwr[:, :, 1] = batch_test1[:, :, 1]
    imwr[:, :, 2] = batch_test1[:, :, 2]
    # axs.imshow(imwr)
    # plt.imshow(imwr)

    imwr1 = imwr.astype(np.float64) / imwr.max()  # normalize the data to 0 - 1
    imwr2 = 255 * imwr1  # Now scale by 255
    img = imwr2.astype(np.uint8)
    imageio.imwrite('res/' + str(filenumber) + '_' + str(c) + '.jpg', img.astype(np.uint8))


def load_data_mat(path, mat2py):
    directory_list = [f for f in os.listdir(path) if not f.startswith('.')]
    Img_train_label = []

    counter = 0

    for i in range(np.size(directory_list)):
        for filename in glob.glob(path + str(directory_list[i]) + '/' + '*.mat'):  # assuming gif

            print(counter)
            mat11 = scipy.io.loadmat(filename)
            mylist = list(mat11.values())
            im1 = mylist[3]
            im = (im1 - np.min(im1)) / np.ptp(im1)

            k1, m1, n1, ch1 = np.shape(im)
            if mat2py == 'True':
                im_n = np.zeros([k1, m1, n1, 1])
                for fr in range(k1):
                    im_n[fr, :, :, :] = np.reshape(im[:, :, fr], [120, 160, 1])

                im = im_n

            print(np.shape(im))
            if counter == 0:
                #   im_ap = np.vstack((im, im))
                sub_s = np.shape(im)[0]
            if counter != 0:
                #   im_ap = np.vstack((im_ap, im))
                counter = counter + 1
            for j in range(np.shape(im)[0]):
                Img_train_label.append(directory_list[i])
        # Img_tr_data.append(np.asarray(im, dtype="float32"))
        # image_list.append(im)

    # im_train = im_ap[sub_s:]
    Img_train_label = np.asarray(Img_train_label, dtype="float32")

    lb = LabelBinarizer()
    lb.fit([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    Img_train_label_f = lb.transform(Img_train_label)
    # return im_train, Img_train_label_f, i+1
    return im, Img_train_label_f, i + 1


def load_data_mat_file(path, directory_list, filename, LV):
    # directory_list = [f for f in os.listdir(path) if not f.startswith('.')]
    Img_train_label = []
    image_train_l_list = []
    im_train_list = []
    im_train_list_align = []
    image_train_l_list_f = []
    counter = 0

    # for i in range(np.size(directory_list)):
    # assuming gif
    Img_train_label = []
    print(counter)
    mat11 = scipy.io.loadmat(filename)
    mylist = list(mat11.values())
    im = mylist[3]
    # im = (im1 - np.min(im1)) / np.ptp(im1) # for normalization


    for j in range(np.shape(im)[0]):
        Img_train_label.append(directory_list)

    image_train_l_list.append(Img_train_label)
    # Img_tr_data.append(np.asarray(im, dtype="float32"))
    # image_list.append(im)

    # im_train = im_ap[sub_s:]
    Img_train_label = np.asarray(Img_train_label, dtype="float32")

    lb = LabelBinarizer()
    lb.fit([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    Img_train_label_f = lb.transform(Img_train_label)

    for j in range(np.shape(image_train_l_list)[0]):
        image_train_l_list_f.append([lb.transform(np.asarray(image_train_l_list[j], dtype="float32"))])

    return im, Img_train_label_f, [im], image_train_l_list_f


def load_data_mat_file1(path, directory_list):
    # directory_list = [f for f in os.listdir(path) if not f.startswith('.')]
    Img_train_label = []
    image_train_l_list = []
    im_train_list = []
    im_train_list_align = []
    image_train_l_list_f = []
    counter = 0

    # for i in range(np.size(directory_list)):
    for filename in glob.glob(path + str(directory_list) + '/' + '*.mat'):  # assuming gif
        Img_train_label = []
        print(counter)
        mat11 = scipy.io.loadmat(filename)
        mylist = list(mat11.values())
        im = mylist[3]
        im_train_list_align.append(im)
        print(np.shape(im))
        im_train_list.append(im)
        if counter == 0:
            im_ap = np.vstack((im, im))
            sub_s = np.shape(im)[0]
        if counter != 0:
            im_ap = np.vstack((im_ap, im))
        counter = counter + 1
        for j in range(np.shape(im)[0]):
            Img_train_label.append(directory_list)

        image_train_l_list.append(Img_train_label)
    # Img_tr_data.append(np.asarray(im, dtype="float32"))
    # image_list.append(im)

    im_train = im_ap[sub_s:]
    Img_train_label = np.asarray(Img_train_label, dtype="float32")

    lb = LabelBinarizer()
    lb.fit([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    Img_train_label_f = lb.transform(Img_train_label)

    for j in range(np.shape(image_train_l_list)[0]):
        image_train_l_list_f.append([lb.transform(np.asarray(image_train_l_list[j], dtype="float32"))])

    return im_train, Img_train_label_f, im_train_list_align, image_train_l_list_f


def align_data(X_test1):
    tr3 = np.zeros([np.shape(X_test1)[2], 120, 160])
    for i in range(np.shape(X_test1)[2]):
        tr = np.reshape(X_test1[:, :, i], [120, 160])
        tr3[i, :, :] = tr
        # print('Test' + str(i))
    print('Aligning finished')
    X_test = tr3
    # my_display(X_test[0, :, :, :])
    return X_test


def align_data2d(X_test1):
    tr3 = np.zeros([np.shape(X_test1)[0], 144, 176, 3])
    for i in range(np.shape(X_test1)[0]):
        tr = np.reshape(X_test1[i, :, :], [176, 144, 3])
        tr1 = np.transpose(tr[:, :, :], [1, 0, 2])
        tr3[i, :, :, :] = tr1
        # print('Test' + str(i))
    print('Aligning finished')
    X_test = tr3
    # my_display(X_test[0, :, :, :])
    return X_test


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


#   ---------------------------------

def CAE_classify(input, kp, name):
    with tf.compat.v1.name_scope(name):
        print('input image1 = ' + str(input.shape))
        input_res = tf.reshape(input, [-1, 132, 144, 1])
        c1 = conv2dr(input_res, name='c1_c', kshape=[5, 5, 1, 32])
        c11 = conv2dr(c1, name='c11_c', kshape=[5, 5, 32, 32])

        p1 = maxpool2d(c11, name='p1_c')

        c21 = conv2dr(p1, name='c12_c', kshape=[5, 5, 32, 64])
        c22 = conv2dr(c21, name='c22_c', kshape=[5, 5, 64, 64])
        p12 = maxpool2d(c22, name='p12_c')

        c31 = conv2dr(p12, name='c31_c', kshape=[3, 3, 64, 128])
        c32 = conv2dr(c31, name='c32_c', kshape=[3, 3, 128, 128])
        p13 = maxpool2d(c32, name='p13_c')

        c41 = conv2dr(p13, name='c41_c', kshape=[3, 3, 128, 128])
        c42 = conv2dr(c41, name='c42_c', kshape=[3, 3, 128, 256])
        p14 = maxpool2d(c42, name='p14_c')

        do1 = dropout(p14, name='do1_C', keep_rate=kp)
        fc3_cls = fullyConnectedf_c(do1, name='fc3_cls', output_size=10)

        return fc3_cls


def CAE_classify_dmd(input, kp, name):
    with tf.compat.v1.name_scope(name):
        print('input image1 = ' + str(input.shape))
        input_res = tf.reshape(input, [-1, 132, 144, 1])
        c1 = conv2dr(input_res, name='c1_c_d', kshape=[5, 5, 1, 32])
        c11 = conv2dr(c1, name='c11_c_d', kshape=[5, 5, 32, 32])

        p1 = maxpool2d(c11, name='p1_c_d')

        c21 = conv2dr(p1, name='c12_c_d', kshape=[5, 5, 32, 64])
        c22 = conv2dr(c21, name='c22_c_d', kshape=[5, 5, 64, 64])
        p12 = maxpool2d(c22, name='p12_c_d')

        c31 = conv2dr(p12, name='c31_c_d', kshape=[3, 3, 64, 128])
        c32 = conv2dr(c31, name='c32_c_d', kshape=[3, 3, 128, 128])
        p13 = maxpool2d(c32, name='p13_c_d')

        c41 = conv2dr(p13, name='c41_c_d', kshape=[3, 3, 128, 128])
        c42 = conv2dr(c41, name='c42_c_d', kshape=[3, 3, 128, 256])
        p14 = maxpool2d(c42, name='p14_c_d')

        do1 = dropout(p14, name='do1_C_d', keep_rate=kp)
        fc3_cls = fullyConnectedf_c(do1, name='fc3_cls_d', output_size=10)

        return fc3_cls


def read_path_LV(path):
    directory_list_ch_test = [f for f in os.listdir(path) if not f.startswith('.')]
    # dire_ch = 'Test/' + directory_list_ch_test[0]
    X_outlier1, y_outlier, n_files_test = load_data_mat(path)

    # X_train_3d = align_data(X_outlier1)
    # X_outlier_2d = np.reshape(X_train_3d, [np.shape(X_train_3d)[0], np.shape(X_train_3d)[1] * np.shape(X_train_3d)[2] * np.shape(X_train_3d)[3]])

    return X_outlier1, y_outlier


def read_path(path, LV):
    directory_list = [f for f in os.listdir(path) if not f.startswith('.')]
    # dire_ch = 'Test/' + directory_list_ch_test[0]
    # X_outlier1, y_outlier, n_files_test = load_data_mat(path, mat2py)
    X_t2 = []
    y_t2 = []
    X_t2s = []
    y_t2s = []
    for k_tc in range(np.size(directory_list)):
        print('----' + str(k_tc))
        filename = glob.glob(path + str(directory_list) + '/' + '*.mat')
        total_files = np.shape(filename)[0]
        for file_num in range(total_files):
            X_t1, y_t1, X_t1s, y_t1s = load_data_mat_file(path, directory_list[k_tc], filename[file_num], LV)

        return X_t1, y_t1, X_t1s, y_t1s


def my_one_class(clf, data):
    pred_data = clf.predict(data)
    sim = pred_data[pred_data == 1].size
    percen_sim = (sim / np.shape(pred_data)[0])
    return percen_sim, pred_data


def proj_kernel(U):
    ntr = len(U)
    ns = ntr
    Kernel = np.eye(ns, ns)

    for n in range(0, ns):
        # print('n:' + str(n))
        if n >= 1:
            U2 = U[n]
            # print('\\nOK : n:'+ str(n))
            for j in range(0, n):
                U1 = U[j]
                # print('j:'+str(j))
                # print(j)\n",
                #           r = np.real(np.trace((U1*np.transpose(U1))*(U2*np.transpose(U2))))\n",
                #           r = np.real((U1*np.transpose(U1)*(U2*np.transpose(U2))))\n",
                # chk = np.dot(U1, np.transpose(U1))
                Um1 = np.dot(U1, np.transpose(np.conj(U1)))
                Um2 = np.dot(U2, np.transpose(np.conj(U2)))
                # Kernel[n, j] = np.real(np.trace(np.dot(Um1, Um2)))
                Kernel[n, j] = np.real(np.dot(Um1, Um2))
                Kernel[j, n] = Kernel[n, j]

    Kernel_n = Kernel

    for k in range(3):
        vmin = np.min(np.real(Kernel_n))
        vmax = np.max(np.real(Kernel_n))
        tmp = (np.real(Kernel_n) - vmin) / (vmax - vmin)
        Kernel_n = tmp
        for ind in range(np.shape(Kernel)[1]):
            Kernel_n[ind, ind] = 1

        # scipy.io.savemat('Kernel_compress_modes.mat', {'Kernel': Kernel})
    return Kernel, Kernel_n


def my_classify_data_read(folder_SVM, folder_path_SVM, LV):  # classify data in folders
    # folder_SVM = 'Test_DMD_modes/'
    # folder_path_SVM = os.path.join(current_path, folder_SVM)

    print('=====')
    X_TRAIN = []
    X_TRAIN_LIST = []
    Y_TRAIN = []
    Y_TRAIN_LIST = []
    j = 0  # counter for stacking data
    directory_list_DMD = sorted([f for f in os.listdir(folder_path_SVM) if not f.startswith('.')])
    for k_tc in range(np.size(directory_list_DMD)):
        print('----' + str(k_tc))
        filenames_DMD = sorted(glob.glob(folder_path_SVM + str(directory_list_DMD[k_tc]) + '/' + '*.mat'))
        total_files = np.shape(filenames_DMD)[0]
        for file_num in range(total_files):
            head, tail = os.path.split(filenames_DMD[file_num])
            tail_name, tail_ext = os.path.splitext(tail)
            X_tr, y_t1, X_l, y_t1s = load_data_mat_file(folder_path_SVM, directory_list_DMD[k_tc],
                                                        filenames_DMD[file_num], LV)
            if j == 0:
                X_TRAIN = X_tr
                Y_TRAIN = y_t1
                X_TRAIN_LIST.append(X_tr)
                Y_TRAIN_LIST.append(y_t1)
            else:
                X_TRAIN = np.vstack((X_TRAIN, X_tr))
                Y_TRAIN = np.vstack((Y_TRAIN, y_t1))
                X_TRAIN_LIST.append(X_tr)
                Y_TRAIN_LIST.append(y_t1)
            j += 1
            print(tail_name)

    return X_TRAIN, Y_TRAIN, X_TRAIN_LIST, Y_TRAIN_LIST


def classify_tr_ts(X_TRAIN, Y_TRAIN, X_TRAIN_LIST, Y_TRAIN_LIST, X_TEST, Y_TEST, TRAIN, TEST):
    if TRAIN == 'True':
        lb = LabelBinarizer()
        lb.fit([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        RY = lb.inverse_transform(Y_TRAIN)
        RX = np.real(X_TRAIN)
        X_TRAIN = []
        Y_TRAIN = []
        clf = svm.NuSVC(gamma='auto')
        num_classes = 2
        num_frames = 47
        full_ind = []
        fval_append = []
        for k in range(np.size(RY)):
            full_ind = np.append(full_ind, k)

        full_ind = full_ind.astype(int)

        train_len = np.shape(X_TRAIN_LIST)[0]
        print('NEW-----')
        avg = []
        for i in range(train_len):
            print('i:', i)
            x_test = X_TRAIN_LIST[i]
            y_test = Y_TRAIN_LIST[i]
            lb.fit([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            y_test = lb.inverse_transform(y_test)

            x_train = []
            y_train = []

            count = 0
            for j in range(train_len):
                if j != i:
                    if count == 0:
                        x_train = X_TRAIN_LIST[j]
                        y_train = Y_TRAIN_LIST[j]
                    else:
                        x_train = np.vstack((x_train, X_TRAIN_LIST[j]))
                        y_train = np.vstack((y_train, Y_TRAIN_LIST[j]))
                    count += 1
            # lb.fit([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            #x_train = np.nan_to_num(x_train)
            Ry = lb.inverse_transform(y_train)
            clf.fit(x_train, Ry)
            predicted = clf.predict(x_test)
            SVM_ACC_TEST = accuracy_score(predicted, y_test)
            print(SVM_ACC_TEST)
            avg = np.append(avg, SVM_ACC_TEST)
        fval = np.mean(avg)
        fval_append.append((fval, fval_append))
        print(fval)
        print('TRAIN')

    if TEST == 'True':
        lb = LabelBinarizer()
        lb.fit([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        RY = lb.inverse_transform(Y_TEST)
        RX = np.real(X_TEST)
        X_TEST = []
        Y_TEST = []
        clf = svm.NuSVC(gamma='auto')
        clf.fit(RX, RY)
        Test_dataX = RX  # [0:22, :]
        predicted = clf.predict(Test_dataX)
        SVM_ACC_TEST = accuracy_score(RY, predicted)

        #  print(predicted)
        #  print('SVM ACCURACY:'+ str(accuracy_score(RY, predicted)))

        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(RX, RY)
        test_data = RX
        # print(np.shape(test_data))
        # test_data1 = np.reshape(test_data[2, :], [1, np.shape(test_data)[1]])
        KNN_pred = neigh.predict(test_data)
        KNN_ACC_TEST = accuracy_score(KNN_pred, KNN_pred)
        print('TESTING........')
    if TEST == 'False':
        SVM_ACC_TEST = 0
        KNN_ACC_TEST = 0

    return fval_append


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def train_network(x):
    # prediction, cost, fc2, Atilde, Phi, S, U, V, W, eigval, x1, x2, lambda1, t_interval, omega = ConvAutoEncoder(x, 'ConvAutoEnc')
    # do3ff, Wenc, Wc1, Wc11, Wc21, Wc22, Wc212, Wc222
    # global fc2oc
    fc4d = ConvAutoEncoderf(x, kp, 'ConvAutoEnc')
    prediction = DeconAutoEncoderf(fc4d, kp, 'DeconAutoEnc')

    with tf.compat.v1.name_scope('cost'):
        # cost = tf.reduce_mean(tf.square(tf.subtract(prediction, x)))
        # tf.compat.v1.reduce_sum(tf.compat.v1.pow(prediction - x, 2))
        cost = tf.compat.v1.reduce_sum(tf.compat.v1.pow(prediction - x, 2))
        # cost = tf.reduce_mean(tf.squared_difference(prediction, x))
        # cross_entropy -tf.reduce_mean(tf.reduce_sum(x * tf.log(decoded), reduction_indices=[1]))
        # cross_entropy = -1. *x *tf.log(prediction) - (1. - x) *tf.log(1. - prediction)
        # cost = tf.reduce_mean(cross_entropy)
    # with tf.name_scope('cost'):
    # cost = tf.reduce_mean(tf.square(tf.subtract(prediction, x)))
    # return output, cost

    # prediction = DeconAutoEncoderf(fc2d, 'DeconAutoEnc')

    # with tf.name_scope('cost'):
    #    cost = tf.reduce_mean(tf.square(tf.subtract(prediction, x)))

    # ----- Classification -----------

    log_my = CAE_classify(fc4d, kp, 'CAE_classify')
    # prediction = DeconAutoEncoderf(fc4d, kp, 'DeconAutoEnc')

    # ---------DMD modes classification --------

    log_DMD = CAE_classify_dmd(DMD_modes, kp, 'DMD_classify')

    loss_dmd_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=log_DMD, labels=Y_cls))
    optimizer_dmd = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_dmd_cls)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=log_my, labels=Y_cls))
    optimizer_c = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    tf_log_my = tf.argmax(log_my, 1)
    tf_Y_cls = tf.argmax(Y_cls, 1)
    correct_pred = tf.equal(tf_log_my, tf_Y_cls)
    accuracy_c = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # ---------- DMD accuracy --------

    tf_log_DMD = tf.argmax(log_DMD, 1)
    tf_Y_cls_DMD = tf.argmax(Y_cls, 1)
    correct_pred_DMD = tf.equal(tf_log_DMD, tf_Y_cls_DMD)
    accuracy_DMD = tf.reduce_mean(tf.cast(correct_pred_DMD, tf.float32))

    # ---------------------------------
    with tf.compat.v1.name_scope('opt'):
        # optimizer1 = tf.train.AdamOptimizer().minimize(cost1)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

    # Create a summary to monitor cost tensor
    tf.compat.v1.summary.scalar("cost", cost)

    # Merge all summaries into a single op
    merged_summary_op = tf.compat.v1.summary.merge_all()

    # ========= Projection Kernel test ==============
    '''
    path_chk = 'Train_DMD/'
    X_train_DMD, X_train_3dm, y_tr_2dm, x_trm, y_trm, X_lm, y_lm = read_path(path_chk, LV='True')

    U = X_lm[0][:]
    Ut = U[:]
    n_cells = len(U)

    # U = 0
    for i in range(n_cells):
        print(' ')
        temp = np.transpose(U[i])[:]
        Ut[i], Sc, Vc = np.linalg.svd(temp, full_matrices=False)


    Kernel, Ker_n = proj_kernel(Ut)


    path_chk = 'Test_DMD/'
    X_test_DMD, X_test_3dm, y_ts_2dm, x_tsm, y_tsm, X_lsm, y_lsm = read_path(path_chk, LV='True')

    Utest = X_lsm[0][:]
    n_cells = len(Utest)
    Ut1 = Utest[:]
    for i in range(n_cells):
        # Ut[i] = np.transpose(U[i])
        temp = np.transpose(Utest[i])[:]
        Ut1[i], Sc, Vc = np.linalg.svd(temp, full_matrices=False)

    Kernel_test, Ker_n_test = proj_kernel(Ut1)



    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.001)
    clf.fit(X_train_DMD)
    y_pred_train_DMD, _ = my_one_class(clf, X_train_DMD)
    y_pred_test_DMD, _ = my_one_class(clf, X_test_DMD)

    '''

    # ============== Load Data LV ===================
    '''
    path_chk = 'Train_LV/'
    X_train_LV, X_train_3d, y_tr_2d, x_tr, y_tr, X_lv, y_lv = read_path(path_chk, LV='True')
    path_chk = 'Test_LV/'
    X_test_LV, X_train_3d, y_tr_2d, x_tr, y_tr, X_lv, y_lv = read_path(path_chk, LV='True')
    # path_chk = 'Outlier_LV/'
    # X_out_LV, y_out_LV = read_path_LV(path_chk)
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.001)
    clf.fit(X_train_LV)
    y_pred_train_LV, _ = my_one_class(clf, X_train_LV)
    y_pred_test_LV, _ = my_one_class(clf, X_test_LV)
    '''
    # ============= Modes Classification==============
    '''
    path_chk = 'Train_DMD/'
    X_train_DMD, X_train_3dm, y_tr_2dm, x_trm, y_trm, X_lm, y_lm = read_path(path_chk, LV='True')
    path_chk = 'Test_DMD/'
    X_test_DMD, X_test_3dm, y_tr_2dm, x_trm, y_trm, X_lm, y_lm = read_path(path_chk, LV='True')
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.001)
    clf.fit(X_train_DMD)
    y_pred_train_DMD, _ = my_one_class(clf, X_train_DMD)
    y_pred_test_DMD, _ = my_one_class(clf, X_test_DMD)
    # y_pred_outliers, _ = my_one_class(clf, X_out_LV)
　　'''
    # ========== Modes + LV ===========================
    '''
    ml_train = np.vstack((X_train_LV, X_train_DMD))
    ml_test = np.vstack((X_test_LV, X_test_DMD))
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.001)
    clf.fit(ml_train)
    y_pred_train_DMD, _ = my_one_class(clf, ml_train)
    y_pred_test_DMD, _ = my_one_class(clf, ml_test)

　　'''

    # --------------- Load Data ----------------------

    # path_chk = 'Train_check/'
    # X_train_2d, X_train_3d, y_tr_2d, x_tr, y_tr, X_l, y_l = read_path(path_chk)
    # path_train_full = '/Users/israr/Desktop/Project_1/Code/Avenue Dataset/training_vol/'
    '''
    path_train_full ='/Users/israr/Desktop/CAEDMD CVIU/Code/train/'
    X_train_2d, X_train_3d, y_train, x_tr, y_tr, X_l1, y_l = read_path(path_train_full, LV='False', mat2py='False')

    X_l = X_l1[:]
    for i in range(np.shape(X_l)[0]):
        X_l[i] = np.reshape(X_l[i], [np.shape(X_l[i])[0], np.shape(X_l[i])[1], np.shape(X_l[i])[2], 1])

    path_test_full = '/Users/israr/Desktop/CAEDMD CVIU/Code/test/'
    X_test_2d, X_test_3d, y_test, x_test, y_test, X_l1_test, y_l_test = read_path(path_train_full, LV='False',
                                                                                  mat2py='False')
    X_l_test = X_l1_test[:]
    for i in range(np.shape(X_l)[0]):
        X_l_test[i] = np.reshape(X_l1_test[i], [np.shape(X_l1_test[i])[0], np.shape(X_l1_test[i])[1], np.shape(X_l1_test[i])[2], 1])
     
     '''

    ''' path_train_full = 'Train_full/'
    X_train_2d, X_train_3d, y_train, x_tr, y_tr, X_l, y_l = read_path(path_train_full, LV='False', mat2py = 'True')

   
    path_train = 'Train_one_svm/'
    X_train_2d, _, _, _, _,_,_ = read_path(path_train, LV='False')
    

    # mat1 = scipy.io.loadmat('Xdata_crp.mat')
    # X_data = mat1['Xdata_crp']
    # X_data = np.reshape(X_data, [150, 28, 28])

    # mat1 = scipy.io.loadmat('ydata_crp.mat')
    # X_label = mat1['ydata_crp']
    path_test = 'Test_one_svm/'
    X_test_2d, _, _, _, _, _, _ = read_path(path_test, LV='False')

#================  Outlier     =================
    path_outlier = 'Outlier_one_svm/'
    X_outlier_2d, _, _, _, _, _, _ = read_path(path_outlier, LV='False')
    
#================ One class SVM ================
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.001)
    clf.fit(X_train_2d[0:500,:])
    y_pred_train, _ = my_one_class(clf, X_train_2d[0:500,:])
    y_pred_test, _ = my_one_class(clf, X_test_2d)
    y_pred_outliers, _ = my_one_class(clf, X_outlier_2d)
   # n_error_train = y_pred_train[y_pred_train == -1].size
   # n_error_test = y_pred_test[y_pred_test == -1].size
   # n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    '''
    # ------------------------------------------------
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [9, 10, 11, 12]])
    # rn2 = np.array([[1,2,3,4,5,6],
    #                [1,2,3,4,5,6],
    #                [1,2,3,4,5,6]])
    u, s, vh = np.linalg.svd(a, full_matrices=True)
    proj_kernel(u)

    n_epochs = 30
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        LV = 'False'
        #path_train = '/Users/israr/Desktop/CAEDMD CVIU/Code/train_chk/'
        #path_test = '/Users/israr/Desktop/CAEDMD CVIU/Code/test_chk/'

        path_train = '/Volumes/LaCie/UCF-tt/train/' # Training folder
        path_test = '/Volumes/LaCie/UCF-tt/test/'  # Testing folder

        current_path = os.getcwd()
        folder_DMD = 'Train_DMD_modes'
        folder_path_DMD = os.path.join(current_path, folder_DMD)
        if not os.path.exists(folder_path_DMD):
            os.mkdir(folder_path_DMD)

        folder_LV = 'Train_LV'
        folder_path_LV = os.path.join(current_path, folder_LV)
        if not os.path.exists(folder_path_LV):
            os.mkdir(folder_path_LV)

        folder_kernel = 'Train_Kernel-U'
        folder_path_kernel = os.path.join(current_path, folder_kernel)
        if not os.path.exists(folder_path_kernel):
            os.mkdir(folder_path_kernel)

        folder_video = 'Train_Video-U'
        folder_path_video = os.path.join(current_path, folder_video)
        if not os.path.exists(folder_path_video):
            os.mkdir(folder_path_video)

        # =============== Testing folders ==================
        folder_DMD_test = 'Test_DMD_modes'
        folder_path_DMD_test = os.path.join(current_path, folder_DMD_test)
        if not os.path.exists(folder_path_DMD_test):
            os.mkdir(folder_path_DMD_test)
        folder_LV_test = 'Test_LV'
        folder_path_LV_test = os.path.join(current_path, folder_LV_test)
        if not os.path.exists(folder_path_LV_test):
            os.mkdir(folder_path_LV_test)
        folder_kernel_test = 'Test_Kernel-U'
        folder_path_kernel_test = os.path.join(current_path, folder_kernel_test)
        if not os.path.exists(folder_path_kernel_test):
            os.mkdir(folder_path_kernel_test)

        # ===================================================

        dmd = DMD()  # DMD initialize
        alist = []
        avg_list1 = []
        avg_costtestlist = []
        count_classify_test = 0
        count_classify_train = 0
        # fig = plt.figure()
        # ax1 = fig.add_subplot(1, 1, 1)
        # batch_x = 20*np.random.rand(149,2500,3)
        for epoch in range(n_epochs):
            countbatch = 0
            avg_cost = 0
            avg_costtest = 0
            # n_batches = int(mnist.train.num_examples / batch_size)
            n_batches = 10
            # rn = random.sample(range(0, 5679), 5679)
            # batch_xr = batch_x1[rn, :, :, :]

            # Loop over all batches
            for ii in range(1):
                # batch_x = batch_xr[ i * batch_size:batch_size * (i + 1), :, :, :]
                # batch_x, batch_y = next_batch(batch_size, X_train_3d, y_train)
                # mini_batch_mean = batch_x.sum(axis=0)/len(batch_x)
                # mini_batch_var = ((batch_x-mini_batch_mean)**2).sum(axis=0)/len(batch_x)
                # batch_xn = (batch_x-mini_batch_mean)/((mini_batch_var + 1e-8) **0.5)

                # batch_x = batch_x1
                # batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                #  _, c, summary, vv, pred, Atildeo, Phio, So, Uo, Vo, Wo, eigvalo , x1o, x2o, omegao, lambda1o = sess.run([optimizer, cost, merged_summary_op, fc2, prediction, Atilde, Phi, S, U, V, W, eigval, x1, x2, omega, lambda1],
                #                                    feed_dict={x: batch_x, y: batch_y})
                # fc2oc = sess.run(fc2, feed_dict={x: batch_x})
                # chk = dmdm(fc2oc)
                # chkt = np.transpose(chk)
                # _, c, summary, pred = sess.run([optimizer, cost, merged_summary_op, prediction], feed_dict={x: batch_x})

                # =======    Training Video Frames      ======
                # _, c, latentvector, outputdec = sess.run([optimizer, cost, fc4d, prediction], feed_dict={x: batch_x, kp: 0.5})

                directory_list = sorted([f for f in os.listdir(path_train) if not f.startswith('.')])
                for k_tc in range(np.size(directory_list)):
                    print('----' + str(k_tc))
                    filename = sorted(glob.glob(path_train + str(directory_list[k_tc]) + '/' + '*.mat'))
                    total_files = np.shape(filename)[0]
                    mse_arr = []

                    # Make folder to save DMD-Modes
                    folder_in_path = os.path.join(folder_path_DMD, str(directory_list[k_tc]))
                    if not os.path.exists(folder_in_path):
                        os.mkdir(folder_in_path)

                    folder_in_path_LV = os.path.join(folder_LV, str(directory_list[k_tc]))
                    if not os.path.exists(folder_in_path_LV):
                        os.mkdir(folder_in_path_LV)

                    folder_in_path_Kernel = os.path.join(folder_kernel, str(directory_list[k_tc]))
                    if not os.path.exists(folder_in_path_Kernel):
                        os.mkdir(folder_in_path_Kernel)

                    folder_in_path_video = os.path.join(folder_video, str(directory_list[k_tc]))
                    if not os.path.exists(folder_in_path_video):
                        os.mkdir(folder_in_path_video)

                    for file_num in range(total_files):
                        head, tail = os.path.split(filename[file_num])
                        tail_name, tail_ext = os.path.splitext(tail)
                        X_tr, y_t1, X_l, y_t1s = load_data_mat_file(path_train, directory_list[k_tc],
                                                                    filename[file_num], LV)

                        # frame_len = 16
                        # nof_d = np.shape(X_l[i])[0]
                        # print(np.floor(nof_d/frame_len ))
                        # loop_d = int(np.round(nof_d / frame_len))
                        # rem = nof_d - loop_d * frame_len
                        mse_arr = []

                        folder_SVM = 'Train_LV-100/'
                        folder_path_SVM = os.path.join(current_path, folder_SVM)
                        X_TRAIN, Y_TRAIN, X_TRAIN_LIST, Y_TRAIN_LIST = my_classify_data_read(folder_SVM,
                                                                                             folder_path_SVM, LV)
                        ACC_List = classify_tr_ts(X_TRAIN, Y_TRAIN, X_TRAIN_LIST, Y_TRAIN_LIST, 0, 0, 'True',
                                                  'False')

                        for j in range(1):  # train a batch 10 times for a video
                            vid_f, vid_label = next_batch(16, X_l[0], y_t1s[0][0])
                            # vid_f = X_l[i][j * (frame_len) + 1:(j + 1) * frame_len]

                            # c, lv, outputdec = sess.run([optimizer ,cost, fc4d, prediction], feed_dict={x: vid_f, kp: 0.7})
                            _, c, lv, outputdec = sess.run([optimizer, cost, fc4d, prediction],
                                                           feed_dict={x: vid_f, kp: 0.7})

                            # c, lv, outputdec = sess.run([cost, fc4d, prediction], feed_dict={x: vid_f, kp: 1})
                            # lv1 = dmdmn(np.transpose(lv[0:-1, :]), np.transpose(lv[1:, :]))
                            # lv1 = np.transpose(lv1)
                            # _, c1, lv2, outputdec1 = sess.run([optimizer, cost, fc4d, prediction],
                            #                                  feed_dict={x: vid_f, fc4d: lv1, kp: 0.5})
                            mse_train = (np.square(outputdec - vid_f)).mean()

                            # folder_SVM = 'Train_LV/'
                            # folder_path_SVM = os.path.join(current_path, folder_SVM, )
                            # SVM_out, KNN_out = my_classify(folder_SVM, folder_path_SVM, LV='False')
                            if j % 10 == 0:
                                print('MSE_TRAIN:' + tail_name + ':' + str(mse_train))
                                my_imwrite('res/', 'Training_original.eps', vid_f[0])
                                my_imwrite('res/', 'Training_predicted.eps', outputdec[0]) # save predicted image in res folder
                                # mse_arr.append(5)

                        c, lv, outputdec = sess.run([cost, fc4d, prediction],
                                                    feed_dict={x: X_l[0], kp: 1})

                        mdict = {'outputdec': outputdec}
                        scipy.io.savemat(folder_in_path_video + '/' + tail_name + '.mat', mdict)

                        mdict = {'outputdec': lv}
                        scipy.io.savemat(folder_in_path_LV + '/' + tail_name + '.mat', mdict)

                        nof = np.shape(lv)[0]
                        dmd = DMD(svd_rank=nof)
                        dmd.fit(lv.T)
                        DMDmodes = dmd.modes
                        mdict = {'DMDmodes': DMDmodes.T}
                        scipy.io.savemat(folder_in_path + '/' + tail_name + '.mat', mdict)

                        # u, s, vh = np.linalg.svd(DMDmodes, full_matrices=True)
                        # u11, u22 = proj_kernel(u)
                        # KU = u22[:, 0]
                        # mdict = {'KernelU': KU}
                        # scipy.io.savemat(folder_in_path_Kernel + '/' + tail_name + '.mat', mdict)

            if epoch % 10 == 0:  # for classification accuracy

                LV = 'False'
                current_path = os.getcwd()

                folder_SVM = 'Train_Kernel-U/'
                folder_path_SVM = os.path.join(current_path, folder_SVM)
                X_TRAIN, Y_TRAIN = my_classify_data_read(folder_SVM, folder_path_SVM, LV)
                TR_Kernel_SVM, TR_Kernel_KNN, TS_Kernel_SVM, TS_Kernel_KNN = classify_tr_ts(X_TRAIN, Y_TRAIN, 0, 0,
                                                                                            'True', 'False')

                folder_SVM = 'Train_DMD_modes/'
                folder_path_SVM = os.path.join(current_path, folder_SVM)
                X_TRAIN, Y_TRAIN = my_classify_data_read(folder_SVM, folder_path_SVM, LV)
                TR_DMD_SVM, TR_DMD_KNN, TS_DMD_SVM, TS_DMD_KNN = classify_tr_ts(X_TRAIN, Y_TRAIN, 0, 0, 'True', 'False')

                folder_SVM = 'Train_LV/'
                folder_path_SVM = os.path.join(current_path, folder_SVM)
                X_TRAIN, Y_TRAIN = my_classify_data_read(folder_SVM, folder_path_SVM, LV)
                TR_LV_SVM, TR_LV_KNN, TS_LV_SVM, TS_LV_KNN = classify_tr_ts(X_TRAIN, Y_TRAIN, 0, 0, 'True', 'False')

                if count_classify_train == 0:
                    f = open("Train_Classification_Accuracy.txt", "w+")
                    f.write("SVM LV Accuracy: %d\r\n" % TR_LV_SVM)
                    f.write("KNN LV Accuracy: %d\r\n\n" % TR_LV_KNN)

                    f.write("SVM Kernel Accuracy: %d\r\n" % TR_Kernel_SVM)
                    f.write("KNN Kernel Accuracy: %d\r\n\n" % TR_Kernel_KNN)

                    f.write("SVM DMD Accuracy: %d\r\n" % TR_DMD_SVM)
                    f.write("KNN DMD Accuracy: %d\r\n\n" % TR_DMD_KNN)
                    f.close()
                    count_classify_train = 1
                else:
                    f = open("Train_Classification_Accuracy.txt", "a+")
                    f.write("SVM LV Accuracy: %d\r\n" % TR_LV_SVM)
                    f.write("KNN LV Accuracy: %d\r\n\n" % TR_LV_KNN)
                    f.write("SVM Kernel Accuracy: %d\r\n" % TR_Kernel_SVM)
                    f.write("KNN Kernel Accuracy: %d\r\n\n" % TR_Kernel_KNN)
                    f.write("SVM DMD Accuracy: %d\r\n" % TR_DMD_SVM)
                    f.write("KNN DMD Accuracy: %d\r\n\n" % TR_DMD_KNN)
                    f.close()

            if epoch % 15 == 0:
                current_path = os.getcwd()
                folder_SVM = '/Volumes/LaCie/UCF/test/'
                folder_path_SVM = os.path.join(current_path, folder_SVM)
                print('=====')
                X_TRAIN = []
                Y_TRAIN = []
                j = 0  # counter for stacking data
                LV = 'False'
                directory_list_train_lv_c = sorted([f for f in os.listdir(folder_path_SVM) if not f.startswith('.')])
                for k_tc in range(np.size(directory_list_train_lv_c)):
                    print('----' + directory_list_train_lv_c[k_tc])
                    filename_train_lv_c = sorted(
                        glob.glob(folder_path_SVM + str(directory_list_train_lv_c[k_tc]) + '/' + '*.mat'))
                    total_files = np.shape(filename_train_lv_c)[0]
                    mse_arr = []
                    # Make folder to save DMD-Modes
                    folder_in_path_test = os.path.join(folder_path_DMD_test, str(directory_list[k_tc]))
                    if not os.path.exists(folder_in_path_test):
                        os.mkdir(folder_in_path_test)
                    folder_in_path_LV_test = os.path.join(folder_LV_test, str(directory_list[k_tc]))
                    if not os.path.exists(folder_in_path_LV_test):
                        os.mkdir(folder_in_path_LV_test)
                    folder_in_path_Kernel_test = os.path.join(folder_kernel_test, str(directory_list[k_tc]))
                    if not os.path.exists(folder_in_path_Kernel_test):
                        os.mkdir(folder_in_path_Kernel_test)
                    for file_num in range(total_files):
                        # print(file_num)
                        head, tail = os.path.split(filename_train_lv_c[file_num])
                        tail_name, tail_ext = os.path.splitext(tail)
                        X_tr, y_t1, X_l, y_t1s = load_data_mat_file(folder_path_SVM, directory_list_train_lv_c[k_tc],
                                                                    filename_train_lv_c[file_num], LV)
                        # c, lv, outputdec = sess.run([cost, fc4d, prediction],
                        #                           feed_dict={x: X_l[0], kp: 1})
                        mdict = {'outputdec': lv}
                        scipy.io.savemat(folder_in_path_LV_test + '/' + tail_name + '.mat', mdict)
                        nof = np.shape(lv)[0]
                        dmd = DMD(svd_rank=nof)
                        dmd.fit(lv.T)
                        DMDmodes = dmd.modes
                        mdict = {'DMDmodes': DMDmodes.T}
                        scipy.io.savemat(folder_in_path_test + '/' + tail_name + '.mat', mdict)
                        u, s, vh = np.linalg.svd(DMDmodes, full_matrices=True)
                        u11, u22 = proj_kernel(u)
                        KU = u22[:, 0]
                        mdict = {'KernelU': KU}
                        scipy.io.savemat(folder_in_path_Kernel_test + '/' + tail_name + '.mat', mdict)

                LV = 'False'
                current_path = os.getcwd()
                # --------------------------------- KERNEL -------------------------------------
                folder_SVM = 'Train_Kernel-U/'
                folder_path_SVM = os.path.join(current_path, folder_SVM)
                X_TRAIN, Y_TRAIN = my_classify_data_read(folder_SVM, folder_path_SVM, LV)

                folder_SVM1 = 'Test_Kernel-U/'
                folder_path_SVM1 = os.path.join(current_path, folder_SVM1)
                X_TEST, Y_TEST = my_classify_data_read(folder_SVM1, folder_path_SVM1, LV)

                TR_Kernel_SVM, TR_Kernel_KNN, TS_Kernel_SVM, TS_Kernel_KNN = classify_tr_ts(X_TRAIN, Y_TRAIN, X_TEST,
                                                                                            Y_TEST, 'True', 'True')
                # -------------------------------------------------------------------------------
                # ---------------------------------DMD Modes-------------------------------------
                folder_SVM = 'Train_DMD_modes/'
                folder_path_SVM = os.path.join(current_path, folder_SVM)
                X_TRAIN, Y_TRAIN = my_classify_data_read(folder_SVM, folder_path_SVM, LV)

                folder_SVM1 = 'Test_DMD_modes/'
                folder_path_SVM1 = os.path.join(current_path, folder_SVM1)
                X_TEST, Y_TEST = my_classify_data_read(folder_SVM1, folder_path_SVM1, LV)

                TR_DMD_SVM, TR_DMD_KNN, TS_DMD_SVM, TS_DMD_KNN = classify_tr_ts(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
                                                                                'True', 'True')

                # --------------------------------------------------------------------------------
                # ------------------------------------LV -----------------------------------------

                folder_SVM = 'Train_LV/'
                folder_path_SVM = os.path.join(current_path, folder_SVM)
                X_TRAIN, Y_TRAIN = my_classify_data_read(folder_SVM, folder_path_SVM, LV)

                folder_SVM1 = 'Test_LV/'
                folder_path_SVM1 = os.path.join(current_path, folder_SVM1)
                X_TEST, Y_TEST = my_classify_data_read(folder_SVM1, folder_path_SVM1, LV)

                TR_LV_SVM, TR_LV_KNN, TS_LV_SVM, TS_LV_KNN = classify_tr_ts(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, 'True',
                                                                            'True')

                if count_classify_test == 0:
                    f = open("Test_Classification_Accuracy.txt", "w+")
                    f.write("SVM LV Accuracy: %d\r\n" % TS_LV_SVM)
                    f.write("KNN LV Accuracy: %d\r\n\n" % TS_LV_KNN)

                    f.write("SVM Kernel Accuracy: %d\r\n" % TS_Kernel_SVM)
                    f.write("KNN Kernel Accuracy: %d\r\n\n" % TS_Kernel_KNN)

                    f.write("SVM DMD Accuracy: %d\r\n" % TS_DMD_SVM)
                    f.write("KNN DMD Accuracy: %d\r\n\n" % TS_DMD_KNN)
                    f.close()
                    count_classify_test = 1
                else:
                    f = open("Test_Classification_Accuracy.txt", "a+")
                    f.write("SVM LV Accuracy: %d\r\n" % TS_LV_SVM)
                    f.write("KNN LV Accuracy: %d\r\n\n" % TS_LV_KNN)
                    f.write("SVM Kernel Accuracy: %d\r\n" % TS_Kernel_SVM)
                    f.write("KNN Kernel Accuracy: %d\r\n\n" % TS_Kernel_KNN)
                    f.write("SVM DMD Accuracy: %d\r\n" % TS_DMD_SVM)
                    f.write("KNN DMD Accuracy: %d\r\n\n" % TS_DMD_KNN)
                    f.close()


            # ======= Video Classification Training ======

            # _, loss_tr, acc_tr, out_log, tf_log_m_o,tf_y_o, tf_pred = sess.run([optimizer_c, loss_op, accuracy_c, log_my, tf_log_my, tf_Y_cls, correct_pred], feed_dict={x: batch_x,
            #                                                     Y_cls: batch_y, kp: 0.5})

            # ======= Latent vectors Classification ======

            # ======== DMD modes classification Training and Testing ==========
            #  for dmd_count in range(np.shape(X_tr2)[0]):
            #      for dmd_count2 in range(np.shape(X_tr2[dmd_count])[0]):
            #           enc_d_r          = sess.run(fc4d, feed_dict={x:X_tr2[dmd_count][dmd_count2], kp: 0})
            #           eig_r, phi_r, atilde_r  =  my_dmdmn(np.transpose(enc_d_r[0:-1,:]), np.transpose(enc_d_r[1:,:]))
            #           _, loss_d, acc_d = sess.run([optimizer_dmd, loss_dmd_cls, accuracy_DMD], feed_dict={DMD_modes: phi_r,
            #                                                  Y_cls: y_tr2[dmd_count][dmd_count2][0][0:-1, :], kp: 0.5})

            #  enc_d_test = sess.run(fc4d, feed_dict={x: X_t2[0], kp: 0})
            #  _, phi_test, _ = my_dmdmn(np.transpose(enc_d_test[0:-1, :]), np.transpose(enc_d_test[1:, :]))
            #  acc_d_test = sess.run(accuracy_DMD,
            #                              feed_dict={DMD_modes: phi_test, Y_cls: y_t2[0][0:-1, :], kp: 0})
            #  print('Trainig loss DMD:', str(loss_d))
            #  print('Trainig accuracy DMD:', str(acc_d))
            #  print('Testing accuracy DMD:', str(acc_d_test))
            # =================================================================

            # print("Minibatch Loss= " + \
            #   "{:.4f}".format(loss_tr) + ", Training Accuracy= " + \
            #   "{:.3f}".format(acc_tr))
            # #batch_x12 = X_test.reshape(np.shape(X_test)[0], 28 * 28)
            # print("Testing Accuracy:", sess.run(accuracy_c, feed_dict={x: X_t2[0], Y_cls: y_t2[0], kp: 0}))

            if epoch % 50 == 0:
                '''
                current_path = os.getcwd()
                folder = 'Train_LV'
                folder_path = os.path.join(current_path, folder)
                if not os.path.exists(folder):
                    os.mkdir(folder_path)
                lb = LabelBinarizer()
                lb.fit([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

                for i in range(np.shape(X_l)[0]):
                    folder_label = lb.inverse_transform(np.array([y_l[i][0][0][0, :]]))
                    folder_in_path = os.path.join(folder_path, str(folder_label[0]))
                    if not os.path.exists(folder_in_path):
                        os.mkdir(folder_in_path)
                    for j in range(np.shape(X_l[i])[0]):
                        print('LV' + str(i))
                        c1, LV = sess.run([cost, fc4d],
                                          feed_dict={x: X_l[i][j], kp: 0})
                        scipy.io.savemat(folder_in_path + '/' + str(j) + '_lv.mat', mdict={'vid_lv': LV})
                
                current_path = os.getcwd()
                folder = 'Train_LV_DMD_abs'
                folder_path = os.path.join(current_path, folder)
                if not os.path.exists(folder):
                    os.mkdir(folder_path)
                lb = LabelBinarizer()
                lb.fit([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                for i in range(np.shape(X_l)[0]):
                    folder_label = lb.inverse_transform(np.array([y_l[i][0][0][0, :]]))
                    folder_in_path = os.path.join(folder_path, str(folder_label[0]))
                    if not os.path.exists(folder_in_path):
                        os.mkdir(folder_in_path)
                    for j in range(np.shape(X_l[i])[0]): #X_l contains latent vectors read from folder
                        print('Wrting DMD modes')
                        mt = (np.transpose(X_l[i][j][:, 0:]))
                        m1 = mt[:, 0:-1]
                        m2 = mt[:, 1:]

                        eig, phi, c = my_dmdmn(m1, m2)
                        scipy.io.savemat(folder_in_path + '/' + str(j) + 'modes_lv.mat',
                                         mdict={'modes_lv': np.transpose(np.abs(phi[:, 0:-10]))})
                '''


            # if i % 5 == 0:
            # my_imwrite(144, 192, i, batch_test1[1, :, :])
            # scipy.misc.imsave('res/'+str(i)+'.jpg', np.transpose(np.reshape(pred_test[1, 0:27648, 0], (192, 144))))

            # if epoch == 10 - 1:
            # saver = tf.train.Saver()
            # saver.save(sess, 'dropout/modelsmalllatent.ckpt')
            # break

            if epoch % 2 == 0:
                current_path = os.getcwd()
                model_path = 'models'
                if os._exists(model_path):
                    if epoch == 0:
                        saver = tf.train.Saver()
                        saver.restore(sess, current_path + model_path + '/' + 'model.ckpt')
                else:
                    saver = tf.compat.v1.train.Saver()
                    saver.save(sess, 'models/classify.ckpt')
            # img = (batch_x[5:6, 0:784].reshape([28, 28]))
            # my_display(img)  # dislay image
            # my_display(pred[5:6, 0:784].reshape([28, 28]))
            # Display logs per epoch step

        # saver = tf.train.Saver()
        # saver.save(sess, '/Users/israr/Desktop/Files/Structured Learning/Tensorflow/venv/model.ckpt')

        # saver = tf.train.Saver()
        # saver.restore(sess, '/Users/israr/Desktop/Files/Structured Learning/Tensorflow/venv/model.ckpt')



train_network(x)
