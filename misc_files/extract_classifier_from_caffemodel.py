import numpy as np
import os.path
from os import listdir
import re
import pdb
import sys
### extract the model(classifier) from caffemodel to train the
### regression network
caffe_root = '/home/selfdriving/zhaotiny/SegNet/caffe-segnet-cudnn52/' 			# Change this to the absolute directoy to SegNet Caffe
sys.path.insert(0, caffe_root + 'python')
import caffe
weight_path = '/media/selfdriving/zhaotiny/SegNet/Models/training_pair_reg/' ## weights
feature_path = '/media/selfdriving/zhaotiny/SegNet/Models/small_features/'
model = '/media/selfdriving/zhaotiny/SegNet/Models/segnet_train_6.prototxt' ## prototxt
reg = 'reg2' # reg 1
iter_number = ['2000', '5000'] # number of iteration you care
number_class = 6 #number class

all_weight = [file for file in listdir(weight_path) if file.endswith('.caffemodel')]
caffe.set_mode_gpu()

for i in range(len(all_weight)):
    cur_weight = all_weight[i]
    digits = re.findall(r'\d+', cur_weight)
    sample = digits[0]
    iter = digits[1]
    iters = digits[2]
    if (iters not in iter_number):
        continue

    weights = os.path.join(weight_path, all_weight[i])
    net = caffe.Net(model,weights,caffe.TEST)

    W = net.params['conv1_1_D_f'][0].data[...]
    b = net.params['conv1_1_D_f'][1].data[...]

    W2 = W.reshape((number_class, -1))
    b2 = b[:, np.newaxis]

    F = np.hstack((W2, b2))
    F2 = F
    data = F2.reshape((number_class, 1, 1, -1))
    data = data.astype('float32')

    for i in range(number_class):
        feature_name = str(number_class) + '_' + str(i + 1) + '_' + sample + '_' + iter + '_' + \
                        iters + '_' + reg + '.npy'
        np.save(os.path.join(feature_path, feature_name), data[[i]])



