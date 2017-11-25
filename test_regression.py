import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
import pdb
from sklearn.preprocessing import normalize
import tensorflow as tf
caffe_root = '/home/selfdriving/zhaotiny/SegNet/caffe-segnet-cudnn52/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')
import scipy.misc
import caffe
from PIL import Image
from function_module import conv2d
# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--conv', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
parser.add_argument('--list', type=str, required = True)
parser.add_argument('--save', type=str, required = True)
args = parser.parse_args()
caffe.set_mode_gpu()



if not os.path.exists(args.save):
	os.mkdir(args.save)

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)
img_list = []
#with open(args.list, 'r') as f:
f = open(args.list, 'r')
for line in f:
	name = line.strip('\t\n')
	img_list.append(name)
f.close()

# Parameter
height = 480
width = 640
n_channel = 64
NUM_CLASSES = 6;
W = tf.placeholder("float", [3, 3, 64, NUM_CLASSES])
B = tf.placeholder("float", [NUM_CLASSES])
F = tf.placeholder("float", [None, n_channel, height, width])

y_pred_seg_small = conv2d(F, W, B, strides=1)
y_pred_seg_small_prob = tf.nn.softmax(y_pred_seg_small, dim = 1)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
data = np.load(args.conv).item()
convW = data['conv/sparse']['weights']
convB = data['conv/sparse']['biases']

init = tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	for i in range(0, args.iter):

		net.forward()

		#image = net.blobs['data'].data
		#label = net.blobs['label'].data
		#predicted = net.blobs['prob'].data
		features = net.blobs['conv1_2_D'].data
		#weights = net.params['conv1_2_D'][0].data
		#print weights
		#pdb.set_trace()
		#image = np.squeeze(image[0,:,:,:])
		[predicted] = sess.run([y_pred_seg_small_prob],
								feed_dict={F: features, W: convW, B: convB})

		output = np.squeeze(predicted[0,:,:,:])
		ind = np.argmax(output, axis=0)
		data_u8 = np.asarray( ind, dtype="uint8" )
		outimg = Image.fromarray( data_u8, "L" )
		outimg.save( args.save + img_list[i])
	print 'Success!'


