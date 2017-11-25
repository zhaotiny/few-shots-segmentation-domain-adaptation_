import numpy as np
import matplotlib.pyplot as plt2
import os.path
import json
import scipy
import argparse
import math
import pylab
import pdb
from sklearn.preprocessing import normalize
import re
caffe_root = '/home/selfdriving/zhaotiny/SegNet/caffe-segnet-cudnn52/' 			# Change this to the absolute directoy to SegNet Caffe
logs_path = './logs3'
output_dir = './models'
import glob

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import pdb
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from function_module import visualize, prediction_net, cal_loss, regress_net, conv2d, \
    sparse_conv, get_variable, regularization_loss, get_train_var, load, snapshot_npy, conv_caffe_to_tf
from Dataset2 import Dataset2
rng = numpy.random
# Import arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--pair', type=str, required=True)
parser.add_argument('--regmodel', type = str, default= "",  \
                     help='pretrained model')
args = parser.parse_args()

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

if (not os.path.exists(logs_path)):
    os.mkdir(logs_path)
files = glob.glob(logs_path + '/*')
for file in files:
    if (os.path.isfile(file)):
        os.remove(file)
# Parameter
height = 480
width = 640
n_channel = 64
learning_rate = 0.005
num_steps = 50000
batch_size = 6
NUM_CLASSES = batch_size
display_step = 50
# Network Parameters
num_input = 577 #  data input (64 * 3 * 3 + 1)
n_hidden_1 = 1024 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_hidden_3 = 1024 # 3rd layder number of neurous
alpha = 0.01 # alpha for leaky relu
lambda1 = 0.01 # weight decay

# tf Graph input
X = tf.placeholder("float", [batch_size, num_input])
F = tf.placeholder("float", [None, n_channel, height, width])
GT = tf.placeholder("float", [None, 1, height, width])
W2 = tf.placeholder("float", [3, 3, 64, NUM_CLASSES])
B2 = tf.placeholder("float", [NUM_CLASSES])
IMG = tf.placeholder("float", [None, height, width, 3])
LABEL = tf.placeholder("float", [None, 1, height, width])
phase = tf.placeholder(tf.bool, name='phase')
phase2 = tf.placeholder(tf.bool, name='phase2')
global_step = tf.Variable(0, trainable=False)
# decreasing learning rate
lr = tf.train.exponential_decay(learning_rate, global_step,
                                1000, 0.95, staircase=True)
## input data
data = Dataset2(args.pair, val = False, defstat = True, numC = NUM_CLASSES)
max_xtd_out = tf.constant(data.max_xtd_out)
min_xtd_out = tf.constant(data.min_xtd_out)
max_xtd_in = tf.constant(data.max_xtd_in)
min_xtd_in = tf.constant(data.min_xtd_in)

# network architecture
X_norm = (X - min_xtd_in) / (max_xtd_in - min_xtd_in)
w_pred = regress_net(X_norm, phase, num_input, n_hidden_1, n_hidden_2, n_hidden_3)
w_pred_denorm = w_pred * (max_xtd_out - min_xtd_out) + min_xtd_out
y_pred_seg_pred = prediction_net(F, w_pred_denorm)
y_pred_seg_pred_prob = tf.nn.softmax(y_pred_seg_pred, dim = 1)
y_pred_seg_pred_fine = sparse_conv(F, NUM_CLASSES, 3, phase2, "conv")

#define loss and optimizer
# segmentaion loss
loss_seg_op = cal_loss(y_pred_seg_pred_fine, GT, NUM_CLASSES)
#regularization loss
conv_W = get_variable("conv", "sparse", "weights")
loss_reg_op = regularization_loss(w_pred_denorm, conv_W, lambda1)
##total loss
total_loss = loss_reg_op + loss_seg_op

conv_variables = get_train_var("conv")
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(total_loss, global_step=global_step, var_list=conv_variables)


##Create an visualization for showing the segmentation output of validation images
## small sampled output
y_pred_seg_small = prediction_net(F, X)
y_pred_seg_small_prob = tf.nn.softmax(y_pred_seg_small, dim = 1)

## caffe model output
pred_test = conv2d(F, W2, B2)
pred_test_prob = tf.nn.softmax(pred_test, dim = 1)

# visualization
label_vis = tf.py_func(visualize, [LABEL, tf.constant(NUM_CLASSES)],tf.float32)
label_vis = tf.reshape(label_vis, (-1, height, width, 3))

y_pred_seg_pred_vis = tf.py_func(visualize, [y_pred_seg_pred, tf.constant(NUM_CLASSES)],tf.float32)
y_pred_seg_pred_vis = tf.reshape(y_pred_seg_pred_vis, (-1, height, width, 3))

y_pred_seg_small_vis = tf.py_func(visualize, [y_pred_seg_small, tf.constant(NUM_CLASSES)],tf.float32)
y_pred_seg_small_vis = tf.reshape(y_pred_seg_small_vis, (-1, height, width, 3))

pred_test_vis = tf.py_func(visualize, [pred_test, tf.constant(NUM_CLASSES)],tf.float32)
pred_test_vis = tf.reshape(pred_test_vis, (-1, height, width, 3))

y_pred_seg_pred_fine_vis = tf.py_func(visualize, [y_pred_seg_pred_fine, tf.constant(NUM_CLASSES)],tf.float32)
y_pred_seg_pred_fine_vis = tf.reshape(y_pred_seg_pred_fine_vis, (-1, height, width, 3))

## logs
tf.summary.scalar("regression_loss", loss_reg_op)
tf.summary.scalar("seg loss", loss_seg_op)
tf.summary.scalar("total loss", total_loss)
tf.summary.image('a ori image', IMG, max_outputs=1)
tf.summary.image('b label', label_vis, max_outputs=1)
tf.summary.image('c pred', y_pred_seg_pred_vis, max_outputs=1)
tf.summary.image('d fine', y_pred_seg_pred_fine_vis, max_outputs=1)
tf.summary.image('e small', y_pred_seg_small_vis, max_outputs=1)
tf.summary.image('f test', pred_test_vis, max_outputs=1)

merged_summary_op = tf.summary.merge_all()

# Start training
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
init = tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    # Run the initializer
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
    if (args.regmodel != ""):
        load(args.regmodel, sess, ignore_missing=True)

    convW = net.params['conv1_1_D_f'][0].data[...]
    convB = net.params['conv1_1_D_f'][1].data[...]

    small_W = np.reshape(convW, (NUM_CLASSES, -1))
    small_B = convB[:, np.newaxis]
    small_weights = np.hstack((small_W, small_B))
    convW = conv_caffe_to_tf(convW, NUM_CLASSES)

    for step in range(1, num_steps+1):

        net.forward()
        image = net.blobs['data'].data
        features = net.blobs['conv1_2_D'].data
        label = net.blobs['label'].data
        predicted = net.blobs['prob'].data

        image2 = image[:, [2, 1, 0], :,:]
        image2 = np.transpose(image2, (0, 2, 3, 1)) / 255.0
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: small_weights, F: features, GT: label, phase2: True, phase: False})

        if  step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
           # [pred, small, test_tf, summary] = sess.run([y_pred_seg_pred_prob, y_pred_seg_small_prob, pred_test_prob, merged_summary_op], \
            #                feed_dict={X: small_W, F: features, GT: label, phase: False, phase2: False, W2: convW, B2: convB, IMG: image2, LABEL: label2})


            [regl, segl, tl, lr2, summary] = sess.run([loss_reg_op, loss_seg_op, total_loss, lr, merged_summary_op],
                            feed_dict={X: small_weights, F: features, GT: label, phase: False, phase2: False, W2: convW, B2: convB, IMG: image2, LABEL: label})

            summary_writer.add_summary(summary, step)

            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.5f} + {:.5f} = {:.5f}, learning rate: {:.5f}".format(regl, segl, tl, lr2))

        if (step % 10000 == 0):
            snapshot_npy(sess, output_dir, "final_", step)
           # pdb.set_trace()
            '''
            pred = visualize(pred)
            large = visualize(large)
            small = visualize(small)
            test_tf = visualize(test_tf)
            test_caffe = visualize(predicted)

            gt = visualize(label)
            image = np.squeeze(image[0,:,:,:])
            image = image / 255.0
            image = np.transpose(image, (1,2,0))
            image = image[:,:,(2,1,0)]
       #     pdb.set_trace()
            plt2.figure()
            plt2.imshow(image,vmin=0, vmax=1)
            plt2.title("original image")
            plt2.figure()
            plt2.imshow(gt,vmin=0, vmax=1)
            plt2.title("Ground-truth")
            plt2.figure()
            plt2.imshow(pred, vmin = 0, vmax = 1)
            plt2.title("Prediction from network")
            plt2.figure()
            plt2.imshow(large,vmin=0, vmax=1)
            plt2.title("Large Weights")
            plt2.figure()
            plt2.imshow(small,vmin=0, vmax=1)
            plt2.title("Small Weights")

            plt2.figure()
            plt2.imshow(test_tf,vmin=0, vmax=1)
            plt2.title("TF Weights")
        #    plt2.figure()
        #    plt2.imshow(test_caffe,vmin=0, vmax=1)
        #    plt2.title("caffe Weights")
            plt2.show()
            '''




#snapshot_npy(sess, num_steps)
    print("Optimization Finished!")





