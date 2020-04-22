import numpy as np
from collections import defaultdict
from utils.constant import START_SYMBOL
import torch
from utils.time_util import current_timestamp


class FSA(object):
    def __init__(self, l1_traces, sentences, k, vob, rnn, partitioner):
        """
        :param l1_traces:
        :param sentences: this sentences should not include stops words if the dataset is real-world
        :param k:
        :param vob:
        :param rnn:
        :param partitioner:
        """
        self.k = k
        self.l1_traces = self.__fsa_format_L1(l1_traces)
        self.sentences = sentences
        self.alphabet = vob
        self.states = [START_SYMBOL] + [str(i) for i in range(k)]
        self.rnn = rnn
        self.partitioner = partitioner
        self.trans_func, self.trans_wfunc, self.final_state = self.learn()

    @staticmethod
    def __fsa_format_L1(l1_traces):
        """ convert PFA-format l1 traces to FSA-format traces.
        Parameter.
        ----------
        l1_traces: PFA format l1 traces, e.g. <'S','0','2','3','P'>
        Return:
            FSA format l1 traces, e.g. <'S','0','2','3'>
        """
        new_l1_traces = [l1_trace[:-1] for l1_trace in l1_traces]
        return new_l1_traces

    def _make_trans_matrix(self):
        '''
        let 0 as the init state, so the cluster 0 should be state 1, and cluster 1 should be state 2.
        :param abs_traces_list:
        :param words_traces:
        :return:
        '''
        #####################
        # count word trigger
        ####################
        trans_cnt = defaultdict(int)  # (i,s,j):freq
        for l1_trace, action_trace in zip(self.l1_traces, self.sentences):
            assert len(l1_trace) - len(action_trace) == 1
            current_state = START_SYMBOL
            for sigma, next_state in zip(action_trace, l1_trace[1:]):
                trans_cnt[(current_state, sigma, next_state)] += 1
                current_state = next_state
        #########################
        # make transition matrix
        ########################
        trans_func = defaultdict(dict)
        trans_wfunc = defaultdict(dict)
        for c_state in self.states:
            for sigma in self.alphabet:
                #######################################
                # select the most frequency transition
                ######################################
                selected_next = -1
                max_freq = 0
                for next_state in self.states:
                    key = (c_state, sigma, next_state)
                    if key in trans_cnt:
                        fre = trans_cnt[key]
                        if fre > max_freq:
                            max_freq = fre
                            selected_next = next_state
                if selected_next != -1:
                    trans_func[c_state][sigma] = selected_next
                    trans_wfunc[c_state][sigma] = max_freq
        return dict(trans_func), dict(trans_wfunc)

    def _make_final_states(self):
        final_states = []
        cluster_centers = self.partitioner.cluster_centers_
        for i in range(self.k):
            cluster_center = cluster_centers[i]
            probs = self.rnn.output_pr_dstr(torch.Tensor(cluster_center)).cpu().detach().squeeze().numpy()
            label = np.argmax(probs)
            if label == 1:
                final_states.append(str(i))
        return final_states

    def learn(self):
        print("{} learning...".format(current_timestamp()))
        trans_func, trans_wfunc = self._make_trans_matrix()
        final_state = self._make_final_states()
        return trans_func, trans_wfunc, final_state

    def predict(self, sent):
        is_unspecified = False
        c_state = START_SYMBOL
        for sigma in sent:
            if sigma in self.trans_func[c_state]:
                c_state = self.trans_func[c_state][sigma]
            else:
                is_unspecified = True
                break
        pred = 1 if c_state in self.final_state else 0
        return pred, is_unspecified
