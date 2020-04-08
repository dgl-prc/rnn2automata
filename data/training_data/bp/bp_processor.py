

import torch
import string
import pickle
class BPProcessor(object):

    def __init__(self):
        self.index_map = {key: i + 1 for i, key in enumerate(string.ascii_lowercase)}
        self.index_map[""] = 0
        self.index_map["("] = 27
        self.index_map[")"] = 28

    def sequence2tensor(self, sequence, is_batch=False):
        '''
        use one-hot to encode the sequence
        :param sequences: e.g.,"100010100"
        :param alphabet_size: the length of one-hot. since we take "",i.e.,empty string, into account, so the size of
                                 alphabet is 3.
        :param is_batch:
        :return:
        '''
        alphabet_size = 29  # empty string,a~z, (, )
        len_seq = 1 if sequence == "" else len(sequence) + 1 # add the empty string
        sequence_tensor = torch.zeros(1, len_seq, alphabet_size)
        sequence_tensor[0][0] = torch.zeros(alphabet_size)
        sequence_tensor[0][0][0] = 1
        if len_seq == 1:
            return sequence_tensor
        else:
            for li,ch in enumerate(sequence):
                vector = torch.zeros(alphabet_size)
                vector[self.index_map[ch]] = 1
                sequence_tensor[0][li+1] = vector
        return sequence_tensor

    def load_data(self, data_path):
        refined_train_data = []
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        train_data = zip(data["data"].keys(), data["data"].values())
        train_data = [(i[0], int(i[1])) for i in train_data]
        for i in train_data:
            if i[0] is '': continue
            refined_train_data.append(i)
        return refined_train_data

    def sequence_purifier(self, sequence):
        return sequence



