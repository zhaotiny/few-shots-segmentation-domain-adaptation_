import os
from PIL import Image
from scipy import misc
import numpy as np
import re
PATH = '/home/selfdriving/zhaotiny/SegNet/cityscape/'
GT = 'labelsIdx'
FREQ = 'freq'
image_folder = os.path.join(PATH, GT)
freq_folder = os.path.join(PATH, FREQ)
gt_files =[file for file in os.listdir(image_folder) if file.endswith('.png')]
if (not os.path.exists(freq_folder)):
    os.mkdir(freq_folder)
num_class = 5
for file in gt_files:
    filename = os.path.join(image_folder, file)
    #img = Image.open(filename)
    img = misc.imread(filename)
    save_name = re.sub(GT, FREQ, filename)
    save_name = save_name[:-4] + '.npy'
    freq = np.zeros(num_class)
    for i in range(num_class):
        freq[i] = np.sum(img == i)
    sumVal = np.sum(freq)
    freq = freq / sumVal
    np.save(save_name, freq)

