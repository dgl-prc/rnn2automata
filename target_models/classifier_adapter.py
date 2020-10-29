import numpy as np
from utils.exceptions import *
from target_models.model_training import sent2tensor


class Classifier(object):
    def __init__(self, rnn, rnn_type, input_dim, word2idx, wv_matrix, device):
        self.rnn = rnn.to(device)
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.word2idx = word2idx
        self.wv_matrix = wv_matrix
        self.device = device

    def format_input(self, sent):
        tensor_sequence = sent2tensor(sent, self.input_dim, self.word2idx, self.wv_matrix, self.device)
        if tensor_sequence.shape[1] == 0:
            raise BadInput("empty input")
        tensor_sequence = tensor_sequence.to(self.device)
        return tensor_sequence

    def get_probs(self, sent):
        '''
        :param sent:
        :return:
        '''
        tensor_sequence = self.format_input(sent)
        output, inter_state = self.rnn(tensor_sequence)
        probs = self.rnn.output_pr_dstr(output[0][-1].unsqueeze(0)).cpu().detach().squeeze().numpy()
        return probs

    def get_label(self, sent):
        '''
        :param sent: the sentence, word list
        :return:
        '''
        probs = self.get_probs(sent)
        return np.argmax(probs)
