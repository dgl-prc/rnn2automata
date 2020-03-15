import sys

sys.path.append("../../../")
from level1_abstract.clustering_based import *
from utils.constant import *
from utils.help_func import load_pickle, get_auc
from target_models.model_helper import load_model
from target_models.model_helper import sent2tensor

MIN_PROB = -1e20


def prepare_L1_data(k, model_type, data_type, device, data_source, pt_type):
    input_dim = 300
    #####################
    # load partitioner
    ####################
    L1_abs_folder = getattr(AbstractData.Level1, pt_type.upper())
    L1_abs_folder = getattr(L1_abs_folder, model_type.upper())
    L1_abs_folder = getattr(L1_abs_folder, data_type.upper())

    if pt_type == "km":
        # Legacy issues
        cluster_path = os.path.join(L1_abs_folder, "k={}".format(k), "{}_kmeans.pkl".format(data_source))
    else:
        cluster_path = os.path.join(L1_abs_folder, "k={}".format(k), "{}_partition.pkl".format(data_source))

    partitioner = load_pickle(get_path(cluster_path))

    #############
    # load data
    #############
    raw_data = load_pickle(get_path(getattr(DataPath, data_type.upper()).PROCESSED_DATA))
    adv_data = load_pickle(get_path(getattr(getattr(Application.AEs, data_type.upper()), model_type.upper())))
    wv_matrix = load_pickle(get_path(getattr(DataPath, data_type.upper()).WV_MATRIX))
    word2idx = raw_data["word_to_idx"]
    ############
    # load model
    ############
    model = load_model(model_type, data_type, device)

    ####################
    # extract ori trace
    ###################
    benign_traces = []
    adv_traces = []
    benign_labels = []
    adv_labels = []
    for ele in adv_data:
        idx, adv_x, adv_y = ele
        benign_x, benign_y = raw_data["test_x"][idx], raw_data["test_y"][idx]

        # benign
        benign_tensor = sent2tensor(benign_x, input_dim, word2idx, wv_matrix, device)
        benign_hn_trace, benign_label_trace = model.get_predict_trace(benign_tensor)
        benign_traces.append(benign_hn_trace)
        assert benign_y == benign_label_trace[-1]
        benign_labels.append(benign_y)

        # adv
        adv_tensor = sent2tensor(adv_x, input_dim, word2idx, wv_matrix, device)
        adv_hn_trace, adv_label_trace = model.get_predict_trace(adv_tensor)
        adv_traces.append(adv_hn_trace)
        assert adv_y == adv_label_trace[-1]
        adv_labels.append(adv_y)

    #############################
    # make level1 abstract traces
    #############################
    benign_abs_seqs = level1_abstract(rnn_traces=benign_traces, y_pre=benign_labels, partitioner=partitioner,
                                      partitioner_exists=True)
    adv_abs_seqs = level1_abstract(rnn_traces=adv_traces, y_pre=adv_labels, partitioner=partitioner,
                                   partitioner_exists=True)

    return benign_abs_seqs, adv_abs_seqs


def _get_path_prob(traces, trans_func, trans_wfunc):
    probs = []
    for seq in traces:
        assert seq[0] == START_SYMBOL
        c_id = 1
        acc_prob = 1.0
        for sigma in seq[1:]:
            sigma = str(sigma)
            if sigma not in trans_wfunc[c_id]:
                acc_prob = MIN_PROB
                break
            else:
                acc_prob *= trans_wfunc[c_id][sigma]
                c_id = trans_func[c_id][sigma]
        # normalize
        acc_prob = np.log(acc_prob) / len(seq) if acc_prob != MIN_PROB else acc_prob
        probs.append(acc_prob)
    return probs


def do_detect(benign_traces, adv_traces, k, model_type, data_type, data_source, total_symbols, pt_type):
    # load dfa
    l2_path = getattr(getattr(getattr(AbstractData.Level2, pt_type.upper()), model_type.upper()), data_type.upper())
    tranfunc_path = get_path(
        os.path.join(l2_path, data_source,
                     "{}_{}_k{}_{}_transfunc.pkl".format(model_type, data_type, k, total_symbols)))
    dfa = load_pickle(tranfunc_path)
    trans_func, trans_wfunc = dict(dfa["trans_func"]), dict(dfa["trans_wfunc"])

    # calculate path prob
    benign_prob = _get_path_prob(benign_traces, trans_func, trans_wfunc)
    adv_prob = _get_path_prob(adv_traces, trans_func, trans_wfunc)

    b_break = len(np.where(np.array(benign_prob) == MIN_PROB)[0])
    adv_break = len(np.where(np.array(adv_prob) == MIN_PROB)[0])
    print("b_break:{},adv_break:{}".format(b_break, adv_break))
    auc = get_auc(pos_score=benign_prob, neg_score=adv_prob)
    return auc


if __name__ == '__main__':
    _device = "cpu"
    _model_type = ModelType.LSTM
    _data_type = DateSet.MR
    _data_source = "test"
    _total_symbols = "49056"
    _pt_type = "hc"
    for _k in range(10, 22, 2):
        benign_abs_seqs, adv_abs_seqs = prepare_L1_data(_k, _model_type, _data_type, _device, _data_source, _pt_type)
        auc = do_detect(benign_abs_seqs, adv_abs_seqs, _k, _model_type, _data_type, _data_source, _total_symbols,
                        _pt_type)
        print("k={}, auc:{:.4f}".format(_k, auc))
