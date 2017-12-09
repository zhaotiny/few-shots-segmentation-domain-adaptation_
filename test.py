import numpy as np
import os.path
import argparse

import tensorflow as tf
from PIL import Image
from function_module import conv2d
from Dataset_f2gt import Dataset_f2gt
import ntpath

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--conv', type=str, required=True,
					help = 'the conv weights obtained from '
						   'regression network or learning to learn')
parser.add_argument('--list', type=str, required = True,
					help = 'list of names you want to save')
parser.add_argument('--save', type=str, required = True,
					help = 'path to save the features')
args = parser.parse_args()

if not os.path.exists(args.save):
	os.mkdir(args.save)

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

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
data = np.load(args.conv).item()
convW = data['conv/sparse']['weights']
convB = data['conv/sparse']['biases']

data_f2gt =Dataset_f2gt(args.list, train = False)
iter = data_f2gt.total_size
output_file_tmp = data_f2gt.output_file
output_file = [os.path.join(args.save, ntpath.basename(file)) for file in output_file_tmp]

init = tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto()) as sess:
	for i in range(0, iter):

		features, _ = data_f2gt.next_batch()
		[predicted] = sess.run([y_pred_seg_small_prob],
								feed_dict={F: features, W: convW, B: convB})

		output = np.squeeze(predicted[0,:,:,:])
		ind = np.argmax(output, axis=0)
		data_u8 = np.asarray( ind, dtype="uint8" )
		outimg = Image.fromarray( data_u8, "L" )
		outimg.save( output_file[i])
	print 'Success!'


