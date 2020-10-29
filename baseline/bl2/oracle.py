import torch
import numpy as np
from target_models.classifier_adapter import Classifier
from utils.constant import *
from data.text_utils import is_artificial
from baseline.bl2.my_string import MyString


class Oracle:
    '''
    Since I wrongly defined the get_first_RState(self) in the orginal RNN and do not want to train them again, I just
    use this class to reload the function.
    class Classifier(object):
        def __init__(self, rnn, rnn_type, input_dim, word2idx, wv_matrix, device)
    '''

    def __init__(self, data_type, model_type, rnn, alphabet, rnn_type, input_dim, word2idx, wv_matrix, device):
        self.data_type = data_type
        self.model_type = model_type
        self.rnn = rnn
        self.oracle = Classifier(rnn, rnn_type, input_dim, word2idx, wv_matrix, device)
        self.alphabet = alphabet
        self.device = device
        self.dtype = torch.float32
        num_directions = 2 if self.rnn.i2h.bidirectional else 1
        self.hn_size = self.rnn.i2h.hidden_size
        self.h_shape = (self.rnn.i2h.num_layers * num_directions, 1, self.hn_size)

    def hx2list(self, hx):
        '''
        :param hx: shape (num_layers * num_directions, batch, hidden_size)
        :return: a list of floats
        '''
        hx = hx.squeeze()
        hx = hx.detach().numpy().flatten().tolist()
        return hx

    def list2hx(self, t_list):
        hx = np.array(t_list, dtype='f')
        hx = hx.reshape(self.h_shape)
        hx = torch.from_numpy(hx)
        return hx

    def _classify_rstate(self, hx):
        probs = self.rnn.output_pr_dstr(hx[0][-1].unsqueeze(0)).cpu().detach().squeeze().numpy()
        label = np.argmax(probs)
        return bool(label)

    def get_first_RState(self):
        """
        **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          Defaults to zero if not provided. If the RNN is bidirectional,
          num_directions should be 2, else it should be 1.
        :return:
        """

        # LSTM
        # def as_vec(self):
        #     return reduce(add, [c.value() for c in self.cs] + [h.value() for h in self.hs])

        # GRU
        # def as_vec(self):
        #     return reduce(add, [h.value() for h in self.hs])

        h0 = torch.zeros(self.h_shape, dtype=self.dtype, device=self.device)
        if self.model_type == ModelType.LSTM:
            hx = self.hx2list(h0) + self.hx2list(h0)
        else:
            hx = self.hx2list(h0)
        label = self._classify_rstate(h0)
        return hx, label

    def get_next_RState(self, h_t, word):
        """ get next hidden state and its label。
        Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment.
        Thus the hidden state should be converted into the list type such that it fits the learning code
        Parameters.
        ----------
        h_t: tensor. hidden state
        word:
        :return:
        """
        assert isinstance(word, str)
        if self.model_type == ModelType.LSTM:
            h_t = (self.list2hx(h_t[:self.hn_size]), self.list2hx(h_t[self.hn_size:]))
        else:
            h_t = self.list2hx(h_t)
        input_tensor = self.oracle.format_input([word])
        output, hx = self.rnn.i2h(input_tensor, h_t)
        if self.model_type == ModelType.LSTM:
            hx = self.hx2list(hx[0]) + self.hx2list(hx[1])
        else:
            hx = self.hx2list(hx)
        label = self._classify_rstate(output)
        return hx, label

    def classify_word(self, sent):  # used for oracle
        if isinstance(sent, str):
            if is_artificial(self.data_type):
                if sent == '':
                    _, pred = self.get_first_RState()
                    return pred
                else:
                    sent = [w for w in sent]
            else:
                sent = sent.split()
        else:
            assert isinstance(sent, MyString)
            if len(sent) == 1 and sent.data[0] == "$":
                _, pred = self.get_first_RState()
                return bool(pred)
            if sent.data[0] == "$":
                sent = sent.data[1:]
        pred = self.oracle.get_label(sent)
        return bool(pred)
