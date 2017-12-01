import numpy as np
import  pdb
import utils

class Dataset(object):

    def __init__(self, file, num_classes = 6, train = True, width = 224, height = 224):
       # self.input_file = []
      #  self.output_file = []
        self.index = 0
        self.load_file(file)
        self.train_size = self.total_size
        self.num_classes = num_classes
        self.width = width
        self.height = height
        self.train = train

    def union_shuffled(self):
        if (self.train):
            print "do you actually want to shuffle?"
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
        Y = np.array([], dtype = np.float32).reshape(0, self.num_classes)
        X = np.array([], dtype = np.float32).reshape(0, self.height, self.height, 3)
        index = self.index
        self.index = index + size;

        for i in range(index, self.index):
            prob = np.load(self.output_file[i]).reshape(1, self.num_classes)
            img = utils.load_image(self.input_file[i]).reshape(1, self.height, self.width, 3)
          #  pdb.set_trace()
            Y = np.vstack((Y, prob))
            X = np.vstack((X, img))
        self.index = self.index % self.train_size
        return X, Y

    def next_batch(self, batchsize):
        if (self.index == 0):
            self.union_shuffled()
        if (self.index + batchsize >= self.train_size):
            return self.read_batch(self.train_size - self.index)
        return self.read_batch(batchsize)




