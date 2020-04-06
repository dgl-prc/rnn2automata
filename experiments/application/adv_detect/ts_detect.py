"""
trace similarity based adversary detection
"""
import sys

sys.path.append("../../../")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from experiments.exp_utils import load_dfa
from experiments.application.adv_detect.detect_utils import *
from experiments.application.adv_detect.textbugger.textbugger_attack import TextBugger
from data.text_utils import filter_stop_words


def _get_L2_trace(L1_trace, trans_func, trans_wfunc):
    NA = "NA"  # not accepted by the automata due to unspecified transition
    c_id = 1
    L2_trace = [str(c_id)]
    for sigma in L1_trace[1:]:
        sigma = str(sigma)
        if sigma not in trans_wfunc[c_id]:
            L2_trace.append(NA)
            break
        else:
            next_id = trans_func[c_id][sigma]
            L2_trace.append(str(next_id))
            c_id = next_id
    return L2_trace


def jaccard_index(set1, set2):
    return len(set1 & set2) / len(set1 | set2)


def make_refers_ori(model_type, data_type, device, size=500, model_path=STANDARD_PATH, use_clean=False):
    """
    Parameters.
    -----------
    model_type:
    data_type:
    device:
    size:
    model_path:
    use_clean: True if stop-words are removed; False if stop-words are kept.
    :return:
    """
    input_dim = 300
    # load data
    #############
    raw_data = load_pickle(get_path(getattr(DataPath, data_type.upper()).PROCESSED_DATA))
    wv_matrix = load_pickle(get_path(getattr(DataPath, data_type.upper()).WV_MATRIX))
    word2idx = raw_data["word_to_idx"]
    ############
    # load model
    ############
    model = load_model(model_type, data_type, device, model_path)
    ori_traces = {}
    ori_traces["X_pos"] = []
    ori_traces["X_neg"] = []
    # note the pre-processed data is shuffled.
    for x, y_ture in zip(raw_data["train_x"], raw_data["train_y"]):
        if use_clean:
            x = filter_stop_words(x)
        if len(ori_traces["X_neg"]) < size or len(ori_traces["X_pos"]) < size:
            tensor_sequence = sent2tensor(x, input_dim, word2idx, wv_matrix, device)
            hn_trace, label_trace = model.get_predict_trace(tensor_sequence)
            if y_ture == 0:
                ori_traces["X_neg"].append(hn_trace)
            else:
                assert y_ture == 1
                ori_traces["X_pos"].append(hn_trace)
    return ori_traces


def make_refers_L1(ori_traces, model_type, data_type, pt_type, partitioner):
    if pt_type == PartitionType.KMP:
        rnn = load_model(model_type, data_type, "cpu")
        pos_abs_seqs = level1_abstract(rnn=rnn, rnn_traces=ori_traces["X_pos"], y_pre=[1] * len(ori_traces["X_pos"]),
                                       partitioner=partitioner, partition_type=pt_type,
                                       partitioner_exists=True)
        neg_abs_seqs = level1_abstract(rnn=rnn, rnn_traces=ori_traces["X_neg"], y_pre=[0] * len(ori_traces["X_neg"]),
                                       partitioner=partitioner, partition_type=pt_type,
                                       partitioner_exists=True)
    else:
        pos_abs_seqs = level1_abstract(rnn_traces=ori_traces["X_pos"], y_pre=[1] * len(ori_traces["X_pos"]),
                                       partitioner=partitioner, partition_type=pt_type,
                                       partitioner_exists=True)
        neg_abs_seqs = level1_abstract(rnn_traces=ori_traces["X_neg"], y_pre=[0] * len(ori_traces["X_neg"]),
                                       partitioner=partitioner, partition_type=pt_type,
                                       partitioner_exists=True)
    refers = {'P': pos_abs_seqs, 'N': neg_abs_seqs}
    return refers


def make_refers_L2(refers, trans_func, trans_wfunc):
    pos_l2 = []
    neg_l2 = []
    for seq in refers["P"]:
        pos_l2.append(_get_L2_trace(seq, trans_func, trans_wfunc))
    for seq in refers["N"]:
        neg_l2.append(_get_L2_trace(seq, trans_func, trans_wfunc))
    return {'P': pos_l2, 'N': neg_l2}


def _state_trace_sim(trace1, trace2):
    set1 = set(trace1)
    set2 = set(trace2)
    return jaccard_index(set1, set2)


def _transition_trace_sim(trace1, trace2):
    set1 = set()
    set2 = set()
    for i in range(len(trace1) - 1):
        set1.add("-".join([trace1[i], trace1[i + 1]]))
    for i in range(len(trace2) - 1):
        set2.add("-".join([trace2[i], trace2[i + 1]]))
    return jaccard_index(set1, set2)


def _cal_dist(L1_traces, refers, trans_func, trans_wfunc):
    dist = []
    for benign_L1_trace in L1_traces:
        l2_trace = _get_L2_trace(benign_L1_trace, trans_func, trans_wfunc)
        R = refers[benign_L1_trace[-1]]
        vec1 = []
        vec2 = []
        for r in R:
            feature1 = _state_trace_sim(l2_trace, r)
            feature2 = _transition_trace_sim(l2_trace, r)
            vec1.append(feature1)
            vec2.append(feature2)
        d1 = np.average(vec1)
        d2 = np.average(vec2)
        # d3 = get_path_prob(benign_L1_trace, trans_func, trans_wfunc)
        # dist.append([d1, d2, d3])
        # dist.append([d1, d2])
        dist.append(d2)
    return dist


def get_raw_auc(y_labels, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_score)
    auc = metrics.auc(fpr, tpr)
    return auc


def do_detect(benign_traces, adv_traces, k, model_type, data_type, data_source, total_symbols, pt_type, refers_l1,
              dfa_file_path=STANDARD_PATH):
    # load dfa
    if dfa_file_path == STANDARD_PATH:
        trans_func, trans_wfunc = load_dfa(model_type, data_type, k, total_symbols, data_source, pt_type)
    else:
        dfa = load_pickle(get_path(dfa_file_path))
        trans_func, trans_wfunc = dict(dfa["trans_func"]), dict(dfa["trans_wfunc"])
    # prepare data
    refers = make_refers_L2(refers_l1, trans_func, trans_wfunc)
    dist_b = _cal_dist(benign_traces, refers, trans_func, trans_wfunc)
    dist_a = _cal_dist(adv_traces, refers, trans_func, trans_wfunc)
    # train classifier
    X = dist_b + dist_a
    Y = [0] * len(dist_b) + [1] * len(dist_a)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2020)
    # clf = LogisticRegression(random_state=0).fit(np.array(X_train), y_train)
    # y_probas = clf.predict_proba(np.array(X_test))
    # acc = clf.score(np.array(X_test), y_test)
    clf = LogisticRegression(random_state=0).fit(np.array(X_train).reshape(-1, 1), y_train)
    y_probas = clf.predict_proba(np.array(X_test).reshape(-1, 1))
    acc = clf.score(np.array(X_test).reshape(-1, 1), y_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_probas[:, 1])
    auc = metrics.auc(fpr, tpr)
    return acc, auc


if __name__ == '__main__':
    _use_clean = True
    _device = "cpu"
    _pt_type = PartitionType.KM
    # adv_bug_mod = TextBugger.DELETE_C
    adv_bug_mod = TextBugger.SUB_W
    _model_type = ModelType.LSTM
    _data_type = DateSet.MR
    _model_path = "data/trained_models/lstm/mr/20200329193810/train_acc-0.7892-test_acc-0.7798.pkl"
    _adv_path = "experiments/application/no_stopws/adv_text/lstm-{}.pkl".format(adv_bug_mod)
    partitioner_path = "experiments/application/no_stopws/l1_trace"
    # _data_source = "test"
    # _total_symbols = "49056"
    _data_source = "train"
    # _total_symbols = 196425
    _total_symbols = 107563
    _refer_size = 400
    alpha = 64  #
    # _k = 10
    for _k in range(2, 22, 2):
        _dfa_file_path = "experiments/application/no_stopws/l2_results/lstm_mr_k{}_alpha_64_107563_transfunc.pkl".format(
            _k)
        pt_path = os.path.join(get_path(partitioner_path), "k={}".format(_k), "{}_partition.pkl".format(_data_source))
        #####################
        # load partitioner
        ####################
        if pt_path == STANDARD_PATH:
            partitioner = load_partitioner(_model_type, _data_type, _pt_type, _k, _data_source)
        else:
            partitioner = load_pickle(pt_path)
        refers_ori = make_refers_ori(_model_type, _data_type, _device, _refer_size, _model_path, _use_clean)
        refers_l1 = make_refers_L1(refers_ori, _model_type, _data_type, _pt_type, partitioner)
        benign_abs_seqs, adv_abs_seqs = prepare_L1_data(_model_type, _data_type, _device, partitioner, _pt_type,
                                                        adv_bug_mod, _model_path, _adv_path, _use_clean)
        acc, auc = do_detect(benign_abs_seqs, adv_abs_seqs, _k, _model_type, _data_type, _data_source, _total_symbols,
                             _pt_type, refers_l1, _dfa_file_path)
        print("k={}, acc:{:.4f}, auc:{:.4f}".format(_k, acc, auc))
