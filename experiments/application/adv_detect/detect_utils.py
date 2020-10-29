import sys

sys.path.append("../../../")
from level1_abstract.clustering_based import *
from utils.constant import *
from utils.help_func import load_pickle
from target_models.model_helper import load_model
from target_models.model_helper import sent2tensor
from data.text_utils import filter_stop_words

MIN_PROB = -1e20


def get_path_prob(t_path, trans_func, trans_wfunc):
    assert t_path[0] == START_SYMBOL
    c_id = 1
    acc_prob = 1.0
    l2_trace = [c_id]
    for sigma in t_path[1:]:
        sigma = str(sigma)
        if sigma not in trans_wfunc[c_id]:
            acc_prob = MIN_PROB
            l2_trace.append("T")  # terminate
            break
        else:
            acc_prob *= trans_wfunc[c_id][sigma]
            c_id = trans_func[c_id][sigma]
            l2_trace.append(c_id)
    acc_prob = np.log(acc_prob) / len(t_path) if acc_prob != MIN_PROB else acc_prob
    return acc_prob, l2_trace


def prepare_L1_data(model_type, data_type, device, partitioner, pt_type, bug_mode, model_pth=STANDARD_PATH,
                    adv_path=STANDARD_PATH, use_clean=False):
    if adv_path == STANDARD_PATH:
        adv_path = get_path(getattr(getattr(Application.AEs, data_type.upper()), model_type.upper()).format(bug_mode))
    else:
        adv_path = get_path(adv_path)
    input_dim = 300
    ######################
    # load data and model
    #####################
    raw_data = load_pickle(get_path(getattr(DataPath, data_type.upper()).PROCESSED_DATA))
    adv_data = load_pickle(adv_path)
    wv_matrix = load_pickle(get_path(getattr(DataPath, data_type.upper()).WV_MATRIX))
    word2idx = raw_data["word_to_idx"]
    model = load_model(model_type, data_type, device, model_pth)

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
        if use_clean:
            # note in this case, the adv is derived from the clean text,thus no filters needed.
            benign_x = filter_stop_words(benign_x)

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
    if pt_type == PartitionType.KMP:
        rnn = load_model(model_type, data_type, "cpu")
    else:
        rnn = None
    benign_abs_seqs = level1_abstract(rnn=rnn, rnn_traces=benign_traces, y_pre=benign_labels, partitioner=partitioner,
                                      partitioner_exists=True, partition_type=pt_type)
    adv_abs_seqs = level1_abstract(rnn=rnn, rnn_traces=adv_traces, y_pre=adv_labels, partitioner=partitioner,
                                   partitioner_exists=True, partition_type=pt_type)

    return benign_abs_seqs, adv_abs_seqs
