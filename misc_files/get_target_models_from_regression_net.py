import numpy as np
import os.path
import argparse

##### get the output from the regression network given the models
##### learned from small samples and the regression network
caffe_root = '/home/selfdriving/zhaotiny/SegNet/caffe-segnet-cudnn52/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import pdb
import tensorflow as tf
import numpy
from function_module import regress_net, load,conv_caffe_to_tf
from Dataset2 import Dataset2

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--regmodel', type = str, default= "",  \
                     help='pretrained model')
parser.add_argument('--folder', type=str, required = True)
parser.add_argument('--name', type=str, required = True)

args = parser.parse_args()

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)
if (not os.path.exists(args.folder)):
    os.mkdir(args.folder)

save_name = os.path.join(args.folder, args.name)

# Parameter
height = 480
width = 640
n_channel = 64
batch_size = 6
NUM_CLASSES = batch_size

# Regression Network Parameters
num_input = 577 #  data input (64 * 3 * 3 + 1)
n_hidden_1 = 1024 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_hidden_3 = 1024 # 3rd layder number of neurous
alpha = 0.01 # alpha for leaky relu

# tf Graph input
X = tf.placeholder("float", [batch_size, num_input])
phase = tf.placeholder(tf.bool, name='phase')
global_step = tf.Variable(0, trainable=False)
## input data
data = Dataset2("", val = False, defstat = True, numC = NUM_CLASSES)
max_xtd_out = tf.constant(data.max_xtd_out)
min_xtd_out = tf.constant(data.min_xtd_out)
max_xtd_in = tf.constant(data.max_xtd_in)
min_xtd_in = tf.constant(data.min_xtd_in)

# network architecture
X_norm = (X - min_xtd_in) / (max_xtd_in - min_xtd_in)
w_pred = regress_net(X_norm, phase, num_input, n_hidden_1, n_hidden_2, n_hidden_3)
w_pred_denorm = w_pred * (max_xtd_out - min_xtd_out) + min_xtd_out


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
init = tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    # Run the initializer
    sess.run(init)
    if (args.regmodel != ""):
        load(args.regmodel, sess, ignore_missing=True)

    convW = net.params['conv1_1_D_f'][0].data[...]
    convB = net.params['conv1_1_D_f'][1].data[...]

    small_W = np.reshape(convW, (NUM_CLASSES, -1))
    small_B = convB[:, np.newaxis]
    small_weights = np.hstack((small_W, small_B))
    convW = conv_caffe_to_tf(convW, NUM_CLASSES)


    w_denorm = sess.run(w_pred_denorm, feed_dict={X: small_weights, phase: False})
    W = w_denorm[:, 0: 576]
    B = w_denorm[:, 576]
    W = np.transpose(W, (1, 0))
    W = np.reshape(W, (64, 3, 3, -1))
    W = np.transpose(W, (1, 2, 0, 3))
    data = {}
    data['conv/sparse'] = {}
    data['conv/sparse']['weights'] = W;
    data['conv/sparse']['biases'] = B;
    np.save(save_name, data)






