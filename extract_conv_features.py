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
#### extract the features before the last classifier from caffe model
caffe_root = '/home/selfdriving/zhaotiny/SegNet/caffe-segnet-cudnn52/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                    help = 'the inference prototxt')
parser.add_argument('--weights', type=str, required=True,
                    help = 'the caffe model ')
parser.add_argument('--iter', type=int, required=True,
                    help = 'number of iteration')
parser.add_argument('--list', type=str, required = True,
                    help = 'list of names you want to save')
parser.add_argument('--save', type=str, required = True,
                    help = 'path to save the features')
args = parser.parse_args()
caffe.set_mode_gpu()

if not os.path.exists(args.save):
	os.mkdir(args.save)

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

img_list = []
f = open(args.list, 'r')
for line in f:
	name = line.strip('\t\n')
	img_list.append(name)
f.close()

for i in range(0, args.iter):
    net.forward()
    features = net.blobs['conv1_2_D'].data
    name = img_list[i][:-4] + '.npy'
    os.remove(args.save + name)
    np.save(args.save + name, features)

print 'Success!'


