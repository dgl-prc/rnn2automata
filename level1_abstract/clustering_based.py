import os
import numpy as np
from utils.constant import START_SYMBOL
from level1_abstract.state_partion import *


def get_term_symbol(y_pre):
    if y_pre == 0:
        return 'N'
    elif y_pre == 1:
        return "P"
    else:
        raise Exception("unknown label:{}".format(y_pre))


def _rnn_traces2point(rnn_traces):
    seq_len = []
    input_points = []
    for seq in rnn_traces:
        seq_len.append(len(seq))
        for hn_state in seq:
            input_points.append(hn_state)
    input_points = np.array(input_points)
    return input_points, seq_len


def make_L1_abs_trace(labels, seq_len, y_pre):
    start_p = 0
    abs_seqs = []
    for size, y in zip(seq_len, y_pre):
        abs_trace = labels[start_p:start_p + size]
        term_symbol = get_term_symbol(y)
        abs_trace = [START_SYMBOL] + abs_trace + [term_symbol]
        abs_seqs.append(abs_trace)
        start_p += size
    return abs_seqs


def level1_abstract(**kwargs):
    '''
    Parameters
    -----------
    partitioner_exists: bool, required.
                   whether or not to use an pre-trained partitioner
    rnn_traces:list(list), required.
               the execute trace of each text on RNN
    y_pre:list, required
                the label of each text given by RNN
    k: int, required when 'kmeans_exists' is false.
                number of clusters to form.
    partitioner: the object of sklearn.cluster.KMeans, required when 'kmeans_exists' is True.
            pre-trained kmeans.
    partition_type: str, option:[km|hc], required if partitioner_exists is false
    -------
    Return:
        abs_seqs: list(list).
                  the level1 abstraction of each rnn trance
        kmeans: the object of sklearn.cluster.KMeans, returned onlt when 'kmeans_exists' is False.
    '''

    rnn_traces = kwargs["rnn_traces"]
    y_pre = kwargs["y_pre"]
    if kwargs["partitioner_exists"]:
        partioner = kwargs["partitioner"]
        input_points, seq_len = _rnn_traces2point(rnn_traces)
        labels = list(partioner.predict(input_points))
        abs_seqs = make_L1_abs_trace(labels, seq_len, y_pre)
        return abs_seqs
    else:
        k = kwargs["k"]
        input_points, seq_len = _rnn_traces2point(rnn_traces)
        if kwargs["partition_type"] == "km":
            partitioner = Kmeans(k)
            partitioner.fit(input_points)
        else:
            partitioner = EHCluster(n_clusters=k)
            partitioner.fit(input_points)
        labels = partitioner.get_fit_labels()
        abs_seqs = make_L1_abs_trace(labels, seq_len, y_pre)
        return abs_seqs, partitioner


def save_level1_traces(abs_seqs, output_path):
    directory = os.path.split(output_path)[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(output_path, "wt") as f:
        for seq in abs_seqs:
            line = ",".join([str(ele) for ele in seq])
            f.write(line + "\n")
    print("Saved to {}".format(output_path))
