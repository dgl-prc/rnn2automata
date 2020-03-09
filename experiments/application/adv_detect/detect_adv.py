from level1_abstract.clustering_based import *
from utils.constant import *
from utils.help_func import load_pickle, get_auc
from target_models.model_helper import load_model
from target_models.model_helper import sent2tensor


def prepare_L1_data(k, model_type, data_type, device):
    input_dim = 300
    ##############
    # load k-means
    ##############
    L1_abs_folder = getattr(getattr(AbstractData.Level1, model_type.upper()), data_type.upper())
    cluster_path = os.path.join(L1_abs_folder, "k={}".format(k), "train_kmeans.pkl")
    kmeans = load_pickle(get_path(cluster_path))

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
    benign_abs_seqs = level1_abstract(rnn_traces=benign_traces, y_pre=benign_labels, kmeans=kmeans, kmeans_exists=True)
    adv_abs_seqs = level1_abstract(rnn_traces=adv_traces, y_pre=adv_labels, kmeans=kmeans, kmeans_exists=True)

    return benign_abs_seqs, adv_abs_seqs


def _get_path_prob(traces, trans_func, trans_wfunc):
    probs = []
    for seq in traces:
        assert seq[0] == START_SYMBOL
        c_id = 1
        acc_prob = 1.0
        for sigma in seq[1:]:
            sigma = str(sigma)
            acc_prob *= trans_wfunc[c_id][sigma]
            c_id = trans_func[c_id][sigma]
        # normalize
        acc_prob = np.log(acc_prob) / len(seq)
        probs.append(acc_prob)
    return probs


def do_detect(benign_traces, adv_traces, k, model_type, data_type):
    # load dfa
    l2_path = getattr(getattr(AbstractData.Level2, model_type.upper()), data_type.upper())
    tranfunc_path = get_path(os.path.join(l2_path, "{}_{}_k{}_187897_transfunc.pkl".format(model_type, data_type, k)))
    dfa = load_pickle(tranfunc_path)  # transfunc[current_id][sigma]=next_id
    trans_func, trans_wfunc = dict(dfa["trans_func"]), dict(dfa["trans_wfunc"])

    # calculate path prob
    benign_prob = _get_path_prob(benign_traces, trans_func, trans_wfunc)
    adv_prob = _get_path_prob(adv_traces, trans_func, trans_wfunc)

    # auc
    auc = get_auc(pos_score=benign_prob, neg_score=adv_prob)

    return auc


if __name__ == '__main__':
    _k = 2
    _device = "cpu"
    _model_type = ModelType.LSTM
    _data_type = DateSet.MR
    benign_abs_seqs, adv_abs_seqs = prepare_L1_data(_k, _model_type, _data_type, _device)
    auc = do_detect(benign_abs_seqs, adv_abs_seqs, _k, _model_type, _data_type)
    print("auc:".format(auc))
