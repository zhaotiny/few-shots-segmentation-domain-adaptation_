import numpy as np
import  pdb
class Dataset(object):

    def __init__(self, file, dimension):
        self.input = np.array([], dtype=np.float32).reshape(0,dimension)
        self.target = np.array([], dtype=np.float32).reshape(0,dimension)
        self.load_file(file)
        self.index = 0

    def unison_shuffled(self):
        p = np.random.permutation(len(self.input))
        self.input = self.input[p]
        self.target = self.target[p]

    def computeStatic(self):
        self.num_std = 3
        self.min_in = np.min(self.input)
        self.max_in = np.max(self.input)
        self.mean_in = np.mean(self.input)
        self.std_in = np.std(self.input)
        self.min_xtd_in = self.mean_in - self.num_std * self.std_in
        self.max_xtd_in = self.mean_in + self.num_std * self.std_in
        self.ratio_in = np.sum(np.logical_and(self.input >= self.min_xtd_in, \
                                self.input <= self.max_xtd_in)) / float(self.input.size)
        print self.min_in, self.min_xtd_in, self.max_in, self.max_xtd_in, self.ratio_in

        self.min_out = np.min(self.target)
        self.max_out = np.max(self.target)
        self.mean_out = np.mean(self.target)
        self.std_out = np.std(self.target)
        self.min_xtd_out = self.mean_out - self.num_std * self.std_out
        self.max_xtd_out = self.mean_out + self.num_std * self.std_out
        self.ratio_out = np.sum(np.logical_and(self.target >= self.min_xtd_out, \
                        self.target <= self.max_xtd_out)) / float(self.target.size)
        print self.min_out, self.min_xtd_out, self.max_out, self.max_xtd_out, self.ratio_out

    def load_file(self, file):
        f = open(file, 'r')

        for line in f:
            files = line.strip('\r\n').split()
            input_file = files[0]
            output_file = files[1]
            small_weight = np.load(input_file)
            large_weight = np.load(output_file)
            small_weight = small_weight.reshape((1, -1))
            large_weight = large_weight.reshape((1, -1))
            self.input = np.vstack((self.input, small_weight))
            self.target = np.vstack((self.target, large_weight))
        f.close()
        self.total_size = self.input.shape[0]
        self.computeStatic()

    ## normalize the data to [0, 1], manually cut the weights if it's outside 3 standard deviation
    def normalize_input(self, input):
        input[input < self.min_xtd_in] = self.min_xtd_in;
        input[input > self.max_xtd_in] = self.max_xtd_in;
        input = (input - self.min_xtd_in) / (self.max_xtd_in - self.min_xtd_in)
        return input

    def denormalize_input(self, input):
        input = input * (self.max_xtd_out - self.min_xtd_in) + self.min_xtd_in
        return input

    def normalize_target(self, target):
        target[target < self.min_xtd_out] = self.min_xtd_out;
        target[target > self.max_xtd_out] = self.max_xtd_out;
        target = (target - self.min_xtd_out) / (self.max_xtd_out - self.min_xtd_out)
        return target

    def denormalize_target(self, target):
        target = target * (self.max_xtd_out - self.min_xtd_out) + self.min_xtd_out
        return target

    def next_batch(self, batchsize):
        if (self.index == 0):
            self.unison_shuffled()
        if (self.index + batchsize >= self.total_size):
            index = self.index
            self.index = 0
            return self.normalize_input(self.input[index:]), \
                   self.normalize_target(self.target[index:])
            #return self.input[index:], self.target[index:]
        index = self.index
        self.index += batchsize
        #return self.input[index: self.index], self.target[index: self.index]
        return self.normalize_input(self.input[index: self.index]), \
               self.normalize_target(self.target[index: self.index])



