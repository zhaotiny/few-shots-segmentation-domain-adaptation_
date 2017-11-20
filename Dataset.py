import numpy as np
import  pdb
   ## sort the classifier pairs to make sure consecutive N pair are from same model
def compare(str11, str22):
    str1 = str11[0]
    str2 = str22[0]
    sptstr1 = str1.split('/')[-1]
    sptstr2 = str2.split('/')[-1]
    sptstr1 = sptstr1[:-4]
    sptstr2 = sptstr2[:-4]

    sptstr1 = sptstr1.split('_')
    sptstr2 = sptstr2.split('_')

    totalC1 = int(sptstr1[0])
    curC1 = int(sptstr1[1])
    numS1 = int(sptstr1[2])
    ithS1 = int(sptstr1[3])
    iter1 = int(sptstr1[4])
    reg1 = int(sptstr1[5][-1])

    totalC2 = int(sptstr2[0])
    curC2 = int(sptstr2[1])
    numS2 = int(sptstr2[2])
    ithS2 = int(sptstr2[3])
    iter2 = int(sptstr2[4])
    reg2 = int(sptstr2[5][-1])
    if (totalC1 == totalC2):
        if (numS1 == numS2):
            if (ithS1 == ithS2):
                if (iter1 == iter2):
                    if (reg1 == reg2):
                        return np.sign(curC1 - curC2)
                    else:
                        return np.sign(reg1 - reg2)
                else:
                    return np.sign(iter1 - iter2)
            else:
                return np.sign(ithS1 - ithS2)
        else:
            return np.sign(numS1 - numS2)
    else:
        return np.sign(totalC1 - totalC2)

def compare2(str11, str22):
    str1 = str11
    str2 = str22
    sptstr1 = str1.split('/')[-1]
    sptstr2 = str2.split('/')[-1]
    sptstr1 = sptstr1[:-4]
    sptstr2 = sptstr2[:-4]

    sptstr1 = sptstr1.split('_')
    sptstr2 = sptstr2.split('_')

    totalC1 = int(sptstr1[0])
    curC1 = int(sptstr1[1])
    numS1 = int(sptstr1[2])
    ithS1 = int(sptstr1[3])
    iter1 = int(sptstr1[4])
    reg1 = int(sptstr1[5][-1])

    totalC2 = int(sptstr2[0])
    curC2 = int(sptstr2[1])
    numS2 = int(sptstr2[2])
    ithS2 = int(sptstr2[3])
    iter2 = int(sptstr2[4])
    reg2 = int(sptstr2[5][-1])
    if (totalC1 == totalC2):
        if (numS1 == numS2):
            if (ithS1 == ithS2):
                if (iter1 == iter2):
                    if (reg1 == reg2):
                        return (curC1 - curC2)
                    else:
                        return (reg1 - reg2)
                else:
                    return (iter1 - iter2)
            else:
                return (ithS1 - ithS2)
        else:
            return (numS1 - numS2)
    else:
        return (totalC1 - totalC2)

class Dataset(object):

    def __init__(self, file, dimension = 577, val = False, defstat = False):
        self.input = np.array([], dtype=np.float32).reshape(0,dimension)
        self.target = np.array([], dtype=np.float32).reshape(0,dimension)
        self.index = 0
        self.val = val;
        self.defstat = defstat
        self.load_file(file)
        if (self.val == True):
            self.train_size = int(0.8 * self.total_size);
        else:
            self.train_size = self.total_size

    def union_shuffled(self):
        p = np.random.permutation(self.train_size)
        self.input[:self.train_size] = self.input[p]
        self.target[:self.train_size] = self.target[p]

    def computeStatic(self):
        self.num_std = 3
        if (self.defstat == True):
            print "use default statistics"
            self.min_in = -0.34493107
            self.max_in = 0.42992684
            self.mean_in = 7.4586083e-06
            self.std_in = 0.05820407

            self.min_out = -1.0602272
            self.max_out = 0.90739739
            self.mean_out = 0.00019841797
            self.std_out = 0.052255213
        else:
            print "use computed statistics"
            self.min_in = np.min(self.input)
            self.max_in = np.max(self.input)
            self.mean_in = np.mean(self.input)
            self.std_in = np.std(self.input)

            self.min_out = np.min(self.target)
            self.max_out = np.max(self.target)
            self.mean_out = np.mean(self.target)
            self.std_out = np.std(self.target)

        self.min_xtd_in = self.mean_in - self.num_std * self.std_in
        self.max_xtd_in = self.mean_in + self.num_std * self.std_in
        self.ratio_in = np.sum(np.logical_and(self.input >= self.min_xtd_in, \
                                self.input <= self.max_xtd_in)) / float(self.input.size)
        print self.min_in, self.min_xtd_in, self.max_in, self.max_xtd_in, self.ratio_in


        self.min_xtd_out = self.mean_out - self.num_std * self.std_out
        self.max_xtd_out = self.mean_out + self.num_std * self.std_out
        self.ratio_out = np.sum(np.logical_and(self.target >= self.min_xtd_out, \
                        self.target <= self.max_xtd_out)) / float(self.target.size)
        print self.min_out, self.min_xtd_out, self.max_out, self.max_xtd_out, self.ratio_out

    def load_file(self, file):
        f = open(file, 'r')
        input_file = []
        output_file = []
        for line in f:
            files = line.strip('\r\n').split()
            input_file.append(files[0])
            output_file.append(files[1])
        f.close()

        tmp = zip(input_file, output_file)
        tmp.sort(compare)
        input_file, output_file = zip(*tmp)

        for i in range(len(input_file)):
            small_weight = np.load(input_file[i])
            large_weight = np.load(output_file[i])
            small_weight = small_weight.reshape((1, -1))
            large_weight = large_weight.reshape((1, -1))
            self.input = np.vstack((self.input, small_weight))
            self.target = np.vstack((self.target, large_weight))

        self.total_size = self.input.shape[0]
        self.computeStatic()

    ## normalize the data to [0, 1], manually cut the weights if it's outside 3 standard deviation
    def normalize_input(self, input):

        input[input < self.min_xtd_in] = self.min_xtd_in;
        input[input > self.max_xtd_in] = self.max_xtd_in;
        input = (input - self.min_xtd_in) / (self.max_xtd_in - self.min_xtd_in)
     #   input = (input - self.mean_in) / self.std_in
        return input

    def denormalize_input(self, input):
        input = input * (self.max_xtd_out - self.min_xtd_in) + self.min_xtd_in
        return input

    def normalize_target(self, target):
        target[target < self.min_xtd_out] = self.min_xtd_out;
        target[target > self.max_xtd_out] = self.max_xtd_out;
        target = (target - self.min_xtd_out) / (self.max_xtd_out - self.min_xtd_out)
       # target = (target - self.mean_out) / self.std_out
        return target

    def denormalize_target(self, target):
        target = target * (self.max_xtd_out - self.min_xtd_out) + self.min_xtd_out
        return target

    def next_batch(self, batchsize):
        if (self.index == 0):
            self.union_shuffled()
        if (self.index + batchsize >= self.train_size):
            index = self.index
            self.index = 0
            return self.normalize_input(self.input[index: self.train_size]), \
                   self.normalize_target(self.target[index: self.train_size])
            #return self.input[index:], self.target[index:]
        index = self.index
        self.index += batchsize
        #return self.input[index: self.index], self.target[index: self.index]
        return self.normalize_input(self.input[index: self.index]), \
               self.normalize_target(self.target[index: self.index])

    def val_set(self):
        return self.normalize_input(self.input[self.train_size:]), \
               self.normalize_target(self.target[self.train_size:])



