import numpy as np
import  pdb
import numpy as np
np.random.seed(0)
#### Provide input pair of features and overall layout distribution for tensorflow
class Dataset_f2ds(object):

    def __init__(self, file, train = True):
        self.index = 0
        self.train = train
        self.load_file(file)
        self.train_size = self.total_size

    def union_shuffled(self):
        if (self.train):
            p = np.random.permutation(self.train_size)
            self.input_file= self.input_file[p]
            self.output_file = self.output_file[p]


    def load_file(self, file):
        f = open(file, 'r')
        input_file = []
        output_file = []
        for line in f:
            files = line.strip('\r\n').split()
            input_file.append(files[0])
            output_file.append(files[1])
        f.close()
        self.total_size = len(input_file)
        self.input_file = np.asarray(input_file)
        self.output_file = np.asarray(output_file)

    def read_batch(self, size):
        index = self.index
        self.index = (index + size);
        features = np.array([])
        prob = np.array([])
        for i in range(index, self.index):
            features_tmp = np.load(self.input_file[i])
            prob_tmp = np.load(self.output_file[i])
            prob_tmp = prob_tmp[np.newaxis, :]
            features = np.vstack([features, features_tmp]) if features.size else features_tmp
            prob = np.vstack([prob, prob_tmp]) if prob.size else prob_tmp
        self.index = (index + size) % self.train_size;
        return features, prob

    def next_batch(self,  batchsize = 1):
        if (self.index == 0):
            self.union_shuffled()
        if (self.index + batchsize >= self.train_size):
            return self.read_batch(self.train_size - self.index)
        return self.read_batch(batchsize)




