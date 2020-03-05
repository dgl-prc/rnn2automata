import numpy as np
from utils.constant import MTYPE_LSTM,MTYPE_GRU
from utils.exceptions import *

class Classifier(object):
    def __init__(self, rnn, dataProcessor,rnn_type):
        self.cuda_no = 0
        self.rnn = rnn.cuda(self.cuda_no)
        self.dataProcessor = dataProcessor
        self.rnn_type = rnn_type

    def get_probs(self, sent):
        '''
        :param sent:
        :return:
        '''
        # if not isinstance(sent,list):
        #     sent = sent.split()
        # if isinstance(sent,unicode):
        #     sent = str(sent)
        if isinstance(sent,list):
            sent = " ".join(sent)
        tensor_sequence = self.dataProcessor.sequence2tensor(sent, 300)
        if tensor_sequence.shape[1] == 0:
            raise BadInput("empty input")
        if self.cuda_no != -1:
            tensor_sequence = tensor_sequence.cuda(self.cuda_no)
        output, inter_state = self.rnn(tensor_sequence) # inter_state is (hn,cn) for LSTM and hn for GRU
        if self.rnn_type == MTYPE_LSTM:
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

