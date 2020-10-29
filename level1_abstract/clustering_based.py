import os
import torch
import numpy as np
from utils.constant import START_SYMBOL, PartitionType
from level1_abstract.state_partion import *
from target_models.model_helper import sent2tensor


def get_term_symbol(y_pre):
    if y_pre == 0:
        return 'N'
    elif y_pre == 1:
        return "P"
    else:
        raise Exception("unknown label:{}".format(y_pre))


def _hn2probas(hn_vec, rnn):
    tensor = torch.unsqueeze(torch.tensor(hn_vec), 0)
    probas = rnn.output_pr_dstr(tensor).cpu().detach().squeeze().numpy()
    return probas


def _rnn_trace2point_probas(rnn_traces, rnn):
    seq_len = []
    input_points = []
    for seq in rnn_traces:
        seq_len.append(len(seq))
        for hn_state in seq:
            probas = _hn2probas(hn_state, rnn)
            input_points.append(probas)
    input_points = np.array(input_points)
    return input_points, seq_len


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
        abs_trace = [START_SYMBOL] + [str(e) for e in abs_trace] + [term_symbol]
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
    partitioner: the object of sklearn.cluster.KMeans, required when 'partitioner_exists' is True.
            pre-trained kmeans.
    partition_type: str, option:[km|km-p|hc], required if partitioner_exists is false
    rnn: rnn model. instance of target_models.my_module.Mymodul.
    -------
    Return:
        abs_seqs: list(list).
                  the level1 abstraction of each rnn trance
        kmeans: the object of sklearn.cluster.KMeans, returned onlt when 'kmeans_exists' is False.
    '''

    rnn_traces = kwargs["rnn_traces"]
    y_pre = kwargs["y_pre"]
    pt_type = kwargs["partition_type"]

    if pt_type == PartitionType.KMP:
        rnn = kwargs["rnn"]
        input_points, seq_len = _rnn_trace2point_probas(rnn_traces, rnn)
    else:
        input_points, seq_len = _rnn_traces2point(rnn_traces)

    if kwargs["partitioner_exists"]:
        partioner = kwargs["partitioner"]
        labels = list(partioner.predict(input_points))
        abs_seqs = make_L1_abs_trace(labels, seq_len, y_pre)
        return abs_seqs
    else:
        k = kwargs["k"]
        if pt_type == PartitionType.HC:
            partitioner = EHCluster(n_clusters=k)
            partitioner.fit(input_points)
        else:
            partitioner = Kmeans(k)
            partitioner.fit(input_points)

        labels = partitioner.get_fit_labels()
        abs_seqs = make_L1_abs_trace(labels, seq_len, y_pre)
        return abs_seqs, partitioner


def sent2L1_trace(sent, model, word2idx, wv_matrix, device, partitioner, pt_type):
    sent_tensor = sent2tensor(sent, 300, word2idx, wv_matrix, device)
    benign_hn_trace, benign_label_trace = model.get_predict_trace(sent_tensor)
    abs_seqs = level1_abstract(rnn_traces=[benign_hn_trace], y_pre=[benign_label_trace[-1]],
                               partitioner=partitioner,
                               partitioner_exists=True, partition_type=pt_type)
    return abs_seqs[0]


def save_level1_traces(abs_seqs, output_path):
    directory = os.path.split(output_path)[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(output_path, "wt") as f:
        for seq in abs_seqs:
            line = ",".join([str(ele) for ele in seq])
            f.write(line + "\n")
    print("Saved to {}".format(output_path))
