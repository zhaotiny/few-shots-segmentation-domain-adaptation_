import numpy as np
import matplotlib.pyplot as plt2
import os.path
import json
import scipy
import argparse
import math
import pylab
import pdb
caffe_root = '/home/selfdriving/zhaotiny/SegNet/caffe-segnet-cudnn52/' 			# Change this to the absolute directoy to SegNet Caffe
logs_path = './logs2'
output_dir = './models'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import pdb
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import glob
from Dataset2 import Dataset2
from function_module import  prediction_net, regress_net, \
    load, snapshot_npy, get_variable, cal_loss, visualize, conv2d, conv_caffe_to_tf
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
learning_rate = 0.001
num_steps = 50000
batch_size = 6
NUM_CLASSES = batch_size
display_step = 400
# Network Parameters
num_input = 577 #  data input (64 * 3 * 3 + 1)
n_hidden_1 = 1024 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_hidden_3 = 1024 # 3rd layder number of neurous
alpha = 0.01 # alpha for leaky relu
lambda1 = 5.0 # ratio between regression loss and seg loss
lambda_min = 1.0

# tf Graph input
X = tf.placeholder("float", [batch_size, num_input])
Y = tf.placeholder("float", [batch_size, num_input])
F = tf.placeholder("float", [None, n_channel, height, width])
GT = tf.placeholder("float", [None, 1, height, width])
W2 = tf.placeholder("float", [3, 3, 64, NUM_CLASSES])
B2 = tf.placeholder("float", [NUM_CLASSES])
IMG = tf.placeholder("float", [None, height, width, 3])
LABEL = tf.placeholder("float", [None, 1, height, width])
phase = tf.placeholder(tf.bool, name='phase')
global_step = tf.Variable(0, trainable=False)
## decreasing weights for regression loss
lambda_tf = tf.train.exponential_decay(lambda1, global_step,
                                       1000, 0.8, staircase = True)
lambda_tf = tf.maximum(tf.constant(lambda_min), lambda_tf)
# decreasing learning rate
lr = tf.train.exponential_decay(learning_rate, global_step,
                                5000, 0.95, staircase=True)
## input data
data = Dataset2(args.pair, val = False, defstat = True, numC = NUM_CLASSES)

# stat from data
max_xtd_out = tf.constant(data.max_xtd_out)
min_xtd_out = tf.constant(data.min_xtd_out)
max_xtd_in = tf.constant(data.max_xtd_in)
min_xtd_in = tf.constant(data.min_xtd_in)

##network architecture
w_pred = regress_net(X, phase, num_input, n_hidden_1, n_hidden_2, n_hidden_3)
w_pred_denorm = w_pred * (max_xtd_out - min_xtd_out) + min_xtd_out
y_pred_seg_pred = prediction_net(F, w_pred_denorm)
y_pred_seg_pred_prob = tf.nn.softmax(y_pred_seg_pred, dim = 1)
w_true = Y

## define loss and optimizer
loss_op = tf.reduce_mean(tf.pow(w_true - w_pred,  2)) #regression loss
loss_seg_op = cal_loss(y_pred_seg_pred, GT, NUM_CLASSES)   ##segmentaion loss
total_loss = lambda_tf * loss_op + loss_seg_op      ##total loss

optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(total_loss, global_step=global_step)

##Create an visualization for showing the segmentation output of validation images
# large sampled output
Y_denorm = Y * (max_xtd_out - min_xtd_out) + min_xtd_out
y_pred_seg_large = prediction_net(F, Y_denorm)
y_pred_seg_large_prob = tf.nn.softmax(y_pred_seg_large, dim = 1)

# small sampled output
X_denorm = X * (max_xtd_out - min_xtd_in) + min_xtd_in
y_pred_seg_small = prediction_net(F, X_denorm)
y_pred_seg_small_prob = tf.nn.softmax(y_pred_seg_small, dim = 1)

# model output
pred_test = conv2d(F, W2, B2)
pred_test_prob = tf.nn.softmax(pred_test, dim = 1)

# visualization
label_vis = tf.py_func(visualize, [LABEL, tf.constant(NUM_CLASSES)],tf.float32)
label_vis = tf.reshape(label_vis, (-1, height, width, 3))

y_pred_seg_pred_vis = tf.py_func(visualize, [y_pred_seg_pred_prob, tf.constant(NUM_CLASSES)],tf.float32)
y_pred_seg_pred_vis = tf.reshape(y_pred_seg_pred_vis, (-1, height, width, 3))

y_pred_seg_large_vis = tf.py_func(visualize, [y_pred_seg_large, tf.constant(NUM_CLASSES)],tf.float32)
y_pred_seg_large_vis = tf.reshape(y_pred_seg_large_vis, (-1, height, width, 3))

y_pred_seg_small_vis = tf.py_func(visualize, [y_pred_seg_small, tf.constant(NUM_CLASSES)],tf.float32)
y_pred_seg_small_vis = tf.reshape(y_pred_seg_small_vis, (-1, height, width, 3))

pred_test_vis = tf.py_func(visualize, [pred_test, tf.constant(NUM_CLASSES)],tf.float32)
pred_test_vis = tf.reshape(pred_test_vis, (-1, height, width, 3))

## logs
tf.summary.scalar("regression_loss", loss_op)
tf.summary.scalar("seg loss", loss_seg_op)
tf.summary.scalar("total loss", total_loss)
tf.summary.histogram('histogram fc1', get_variable('fc1', 'dense', 'weights'))
tf.summary.histogram('histogram fc2', get_variable('fc2', 'dense', 'weights'))
tf.summary.histogram('histogram fc3', get_variable('fc3', 'dense', 'weights'))
tf.summary.histogram('histogram fc4', get_variable('fc4', 'dense', 'weights'))
tf.summary.image('a ori image', IMG, max_outputs=1)
tf.summary.image('b label', label_vis, max_outputs=1)
tf.summary.image('c pred1', y_pred_seg_pred_vis, max_outputs=1)
tf.summary.image('d small', y_pred_seg_small_vis, max_outputs=1)
tf.summary.image('e large', y_pred_seg_large_vis, max_outputs=1)
tf.summary.image('f test', pred_test_vis, max_outputs=1)

merged_summary_op = tf.summary.merge_all()

# Start training
init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    # Run the initializer
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
    if (args.regmodel != ""):
        load(args.regmodel, sess, ignore_missing=True)
    for step in range(1, num_steps+1):
        net.forward()
        image = net.blobs['data'].data
        features = net.blobs['conv1_2_D'].data
        label = net.blobs['label'].data
        predicted = net.blobs['prob'].data
        convW = net.params['conv1_1_D_f'][0].data[...]
        convB = net.params['conv1_1_D_f'][1].data[...]
        convW = conv_caffe_to_tf(convW, NUM_CLASSES)

        image2 = image[:, [2, 1, 0], :,:]
        image2 = np.transpose(image2, (0, 2, 3, 1)) / 255.0

        batch_x, batch_y = data.next_batch()
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, F: features, GT: label, phase: True})


        if  step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            [regl, segl, tl, lr2, lambda_val, summary] = sess.run([loss_op, loss_seg_op, total_loss, lr, lambda_tf, merged_summary_op],
                                    feed_dict={X: batch_x, Y: batch_y, F: features, GT: label, phase: False, W2: convW, B2: convB, IMG: image2, LABEL: label})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.5f} + {:.5f} = {:.5f}, learning rate: {:.5f}, lambda: {:.2f}".format(regl, segl, tl, lr2, lambda_val))

           # [pred, large, small, test_tf, summary] = sess.run([y_pred_seg_pred_prob, y_pred_seg_large_prob, y_pred_seg_small_prob, pred_test_prob, merged_summary_op], \
          #                  feed_dict={X: batch_x, Y: batch_y, F: features, GT: label, phase: False, W2: convW, B2: convB, IMG: image2, LABEL: label})
            summary_writer.add_summary(summary, step)
            if (step % 10000 == 0):
                snapshot_npy(sess, output_dir, "reg2_", step)
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




    print("Optimization Finished!")





