import numpy as np
import  pdb
from scipy import misc
from Dataset_f2ds import Dataset_f2ds
#### Provide input pairs of features and GT segmentation for tensorflow
class Dataset_f2gt(Dataset_f2ds):

    def __init__(self, file, train = True):
        super(Dataset_f2gt, self).__init__(file, train)


    def read_batch(self, size):
        index = self.index
        self.index = (index + size);
        features = np.array([])
        gt = np.array([])
        for i in range(index, self.index):
           # print self.input_file[i], self.output_file[i]
            features_tmp = np.load(self.input_file[index])
            gt_tmp = misc.imread(self.output_file[index])
            gt_tmp = gt_tmp[np.newaxis, np.newaxis, :]
            features = np.vstack([features, features_tmp]) if features.size else features_tmp
            gt = np.vstack([gt, gt_tmp]) if gt.size else gt_tmp
        self.index = (index + size) % self.train_size;
        return features, gt






