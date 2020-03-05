import os
import numpy as np
from sklearn.cluster import KMeans
from utils.constant import START_SYMBOL


def get_term_symbol(y_pre):
    if y_pre == 0:
        return 'N'
    elif y_pre == 1:
        return "P"
    else:
        raise Exception("unknown label:{}".format(y_pre))


def level1_abstract(rnn_traces, y_pre, k):
    '''
    Parameters
    -----------
    rnn_traces:
    ori_data:
    k: number of clusters
    :return:
    '''
    seq_len = []
    input_points = []
    for seq in rnn_traces:
        seq_len.append(len(seq) - 1)
        for hn_state in seq[:-1]:
            input_points.append(hn_state)
    input_points = np.array(input_points)
    kmeans = KMeans(n_clusters=k).fit(input_points)
    labels, cluster_centers = list(kmeans.labels_), kmeans.cluster_centers_
    start_p = 0
    abs_seqs = []
    for size, y in zip(seq_len, y_pre):
        abs_trace = labels[start_p:start_p + size]
        term_symbol = get_term_symbol(y)
        abs_trace = [START_SYMBOL] + abs_trace + [term_symbol]
        abs_seqs.append(abs_trace)
        start_p += size
    return abs_seqs


def save_level1_traces(abs_seqs, output_path):
    directory = os.path.split(output_path)[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(output_path, "wt") as f:
        for seq in abs_seqs:
            line = ",".join([str(ele) for ele in seq])
            f.write(line + "\n")
    print("Saved to {}".format(output_path))
