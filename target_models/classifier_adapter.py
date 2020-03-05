import numpy as np
from utils.constant import *
from utils.exceptions import *
from target_models.model_training import sent2tensor


class Classifier(object):
    def __init__(self, rnn, rnn_type, input_dim, word2idx, wv_matrix, device):
        self.rnn = rnn
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.word2idx = word2idx
        self.wv_matrix = wv_matrix
        self.device = device

    def get_probs(self, sent):
        '''
        :param sent:
        :return:
        '''
        if isinstance(sent, list):
            sent = " ".join(sent)
        tensor_sequence = sent2tensor(sent, 300, self.word2idx, self.input_dim, self.wv_matrix)
        if tensor_sequence.shape[1] == 0:
            raise BadInput("empty input")
        tensor_sequence = tensor_sequence.to(self.device)
        output, inter_state = self.rnn(tensor_sequence)
        # inter_state is (hn,cn) for LSTM and hn for GRU
        if self.rnn_type == ModelType.LSTM:
            last_hn = inter_state[0][-1]
        else:
            last_hn = inter_state[-1]
        probs = self.rnn.output_pr_dstr(last_hn).cpu().detach().squeeze().numpy()
        return probs

    def get_label(self, sent):
        '''
        :param sent: the sentence, word list
        :return:
        '''
        probs = self.get_probs(sent)
        return np.argmax(probs)
