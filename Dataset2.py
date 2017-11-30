import numpy as np
import numpy.matlib
import pdb
from Dataset import Dataset
#### Provide input pairs of small sampled and large sampled models
#### Note that one model might contain several classifiers
class Dataset2(Dataset):
    def __init__(self, file, dimension = 577, val = False, defstat = False, numC = 6):
        super(Dataset2, self).__init__(file, dimension, val, defstat)
        self.numC = numC

    def next_batch(self):
        return super(Dataset2, self).next_batch(self.numC)

    def union_shuffled(self):
        p = np.random.permutation(self.train_size / self.numC)
        p = np.matlib.repmat(p, self.numC, 1)
        p = np.reshape(p, (-1), 'F')
        p2 = np.arange(self.numC)
        p2 = np.matlib.repmat(p2, 1, self.train_size / self.numC)
        p2 = np.squeeze(p2)
        p = self.numC * p + p2
        self.input[:self.train_size] = self.input[p]
        self.target[:self.train_size] = self.target[p]

