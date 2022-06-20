__author__ = 'jasper.zuallaert'
import math
import numpy as np

# Initialize a dictionary to help with getting amino acid ids from sequence
s1 = 'ACDEFGHIKLMNPQRSTVWY'
d_acids = {c:s1.index(c) for c in s1}

# Returns the id for the amino acid ngram supplied
# +1 at the end is because of the zeropadding (0 = empty amino acid)
def get_amino_acid_id(ngram):
    if any(x in ngram for x in 'XOJUBZ'): # DIRTY HARD CODING
        return 0
    num = 0
    for i in range(len(ngram)):
        num += 20**i * d_acids[ngram[-(i+1)]]
    return num

def get_sequences(datafile, max_length):
    all_lines = open(datafile).readlines()

    ids = []
    seqs = []
    labels = []

    for line in all_lines:
        id,seq,label = line.rstrip().split(',')
        ids.append(id)
        labels.append(label)
        seqs.append(seq)

    ids = np.asarray(ids)
    x_data = np.zeros((len(ids),max_length,20),np.float32)
    x_lengths = np.zeros(len(ids),np.int32)
    y_data = np.zeros((len(ids),1),np.int32)

    for i in range(len(ids)):
        y_data[i] = labels[i]
        seq = seqs[i]
        x_lengths[i] = len(seq)
        for j in range(min(max_length,len(seq))):
            x_data[i][j][get_amino_acid_id(seq[j])] = 1

    return Dataset(ids, x_data,x_lengths,y_data)

class Dataset:
    def __init__(self, ids, x_data, x_lengths, y_data):
        self.index_in_epoch = 0
        self.ids = ids
        self.x_data = x_data
        self.x_lengths = x_lengths
        self.y_data = y_data
        self.num_samples = x_data.shape[0]

    # Returns the number of samples in this dataset
    def __len__(self):
        return self.num_samples

    # Returns the maximum sequence length in this dataset
    def get_sequence_length(self):
        return len(self.x_data[0])

    # Returns the x_data, x_lengths, y_data, but only for the samples in the next batch. It also returns
    # a boolean indicating whether the batch returned is the last batch in the dataset (if so, the next call to
    # next_batch will return the first batch of the next epoch)
    def next_batch(self,batch_size):
        start = self.index_in_epoch
        end = self.index_in_epoch + batch_size

        if start == 0:
            idx = np.arange(0, self.num_samples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexes
            self.x_data = self.x_data[idx]
            self.ids = self.ids[idx]
            self.x_lengths = self.x_lengths[idx]
            self.y_data = self.y_data[idx]

        if end < self.num_samples:
            self.index_in_epoch = end
            return self.ids[start:end], self.x_data[start:end], self.x_lengths[start:end], self.y_data[start:end], False # epoch finished = False
        else:
            self.index_in_epoch = 0
            return self.ids[start:], self.x_data[start:], self.x_lengths[start:], self.y_data[start:], True #epoch finished = True

    # Return the amount of steps per epoch, given a batch_size
    def steps_in_epoch(self, batch_size):
        return math.ceil(len(self) / batch_size)

    def get_x(self):
        return self.x_data

    def get_y(self):
        return self.y_data

    def get_lengths(self):
        return self.x_lengths

    # ONLY USE WHEN ONLY ONE CLASS PRESENT IN DATASET
    # Returns the amount of positive samples in the dataset
    def get_positive_count(self):
        assert len(self.y_data[0]) == 1
        return int(np.sum(self.y_data))

    # ONLY USE WHEN ONLY ONE CLASS PRESENT IN DATASET
    # Returns the amount of negative samples in the dataset
    def get_negative_count(self):
        assert len(self.y_data[0]) == 1
        return int(len(self.y_data) - np.sum(self.y_data))