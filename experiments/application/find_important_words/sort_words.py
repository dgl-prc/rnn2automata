import sys
sys.path.append('../../../')
from level1_abstract.clustering_based import *
from level2_abstract.read_seq import *
from level2_abstract.do_L2_abstract import get_L1_data_path
from experiments.exp_utils import *
from experiments.application.find_important_words.get_reachability_matrix import reachability_matrix


def sort_words_by_rm(sent, l1_trace, reach_matrix, trans_func):
    '''sort words by reachability matrix
    Parameters
    ----------
    sent: list, in which each element is a word
    l1_trace: list, in which each element is a level1 abstract state.
    reach_matrix: ndarray, shape(num_states, num_classes). the reachability matrix, first column of which is negative
                  and the second is positive.
    Return:
        rank_list: list, in which each element is word and all words are sorted by the oder of significance.
    '''
    # add start symbol S and the predict label [P|N]
    assert len(l1_trace) - len(sent) == 2, print(len(l1_trace),len(sent))
    c_id = 1  # current state id
    y_predict = l1_trace[-1]
    score_list = []
    for sigma, word in zip(l1_trace[1:-1], sent):
        sigma = str(sigma)
        if sigma not in trans_func[c_id]:
            continue
        else:
            next_id = trans_func[c_id][sigma]
            reach_probs = reach_matrix[next_id - 1]
            if y_predict == "P":
                if reach_probs[1] > reach_probs[0]:
                    score_list.append(reach_probs[1])
                else:
                    score_list.append(-1)
            else:
                assert y_predict == "N"
                if reach_probs[0] > reach_probs[1]:
                    score_list.append(reach_probs[0])
                else:
                    score_list.append(-1)
    sort_idxs = np.argsort(score_list)[::-1]
    sorted_words = [sent[idx] for idx in sort_idxs if score_list[idx] != -1]
    return sorted_words


def main1(**kwargs):
    model_type = kwargs["m_tp"]
    data_type = kwargs["d_tp"]
    data_source = kwargs["d_s"]
    pt_type = kwargs["pt_tp"]
    total_symbols = kwargs["t_syms"]
    k = kwargs["k"]
    num_prop = kwargs["num_prop"]

    pm_file = getattr(getattr(getattr(AbstractData.Level2, pt_type.upper()), model_type.upper()), data_type.upper())
    pm_file = os.path.join(get_path(pm_file), data_source,
                           "{}_{}_k{}_{}.pm".format(model_type, data_type, k, total_symbols))

    trans_func, trans_wfunc = load_dfa(model_type, data_type, k, total_symbols, data_source, pt_type)
    reach_matrix = reachability_matrix(pm_file, num_prop)
    raw_data = load_pickle(get_path(getattr(DataPath, data_type.upper()).PROCESSED_DATA))
    data_path = get_L1_data_path(pt_type, model_type, data_type, data_source, k)
    sequences, alphabet = load_trace_data(data_path, total_symbols)

    cnt1 = 0
    cnt2 = 0
    num_empty = 0
    num_error = 0

    for sent, y_true, l1_trace in zip(raw_data["test_x"], raw_data["test_y"], sequences):
        ordered_sent = sort_words_by_rm(sent, l1_trace, reach_matrix, trans_func)
        print("=================================")
        print("Ori: {}".format(" ".join(sent)))
        print("Ord: {}".format(" ".join(ordered_sent)))

        if len(ordered_sent) == 0:
            num_empty += 1
            if (y_true == 1 and l1_trace[-1] == "N") or (y_true == 0 and l1_trace[-1] == "P"):
                cnt1 += 1

        if (y_true == 1 and l1_trace[-1] == "N") or (y_true == 0 and l1_trace[-1] == "P"):
            num_error += 1
            if len(ordered_sent) == 0:
                cnt2 += 1
    print("Total test samples:{}".format(len(raw_data["test_x"])))
    print("Total {} samples failed to get ordered, {:.4f} of these are wrongly predicted".format(num_empty,
                                                                                                 cnt1 / num_empty))
    print(
        "Total {} samples wrongly predicted, {:.4f} of these failed to get ordered".format(num_error, cnt2 / num_error))

if __name__ == '__main__':
    # _pm_file = "/home/dgl/project/learn_automata_rnn/data/level2_results/km/lstm/test/lstm_mr_k4_49056.pm"
    # rm = reachability_matrix(_pm_file, _num_prop)
    _num_prop = 2
    _k = 2
    _model_type = ModelType.LSTM
    _data_type = DateSet.MR
    _data_source = "train"
    _total_symbols = 196425
    # _data_source = "test"
    # _total_symbols = 49056
    _pt_type = PartitionType.KM

    main1(m_tp=_model_type, d_tp=_data_type, d_s=_data_source, pt_tp=_pt_type, t_syms=_total_symbols, k=_k,
         num_prop=_num_prop)
