import sys
import math
import copy

sys.path.append('../../../')
from level1_abstract.clustering_based import *
from level2_abstract.read_seq import *
from level2_abstract.do_L2_abstract import get_L1_data_path
from experiments.exp_utils import *
from experiments.application.find_important_words.get_reachability_matrix import reachability_matrix
from experiments.application.adv_detect.pprobs_detect import get_path_prob


def get_reach_probs_list(l1_trace, trans_func, reach_matrix):
    c_id = 1  # current state id
    reach_probs_list = []
    for sigma in l1_trace[1:-1]:
        sigma = str(sigma)
        if sigma not in trans_func[c_id]:
            break
        else:
            next_id = trans_func[c_id][sigma]
            reach_probs = reach_matrix[next_id - 1]
            reach_probs_list.append(reach_probs)
    return reach_probs_list


def get_word_score_by_ration(desired_y, reach_probs):
    eps = 0.0000001  # to avoid divided by zero
    if desired_y == "P":
        return reach_probs[1] / (reach_probs[0] + eps)
    else:
        assert desired_y == "N"
        return reach_probs[0] / (reach_probs[1] + eps)


def get_word_score_by_significance(desired_y, pre_reach_probs, cur_reach_probs, position=0):
    # ***Type1: only use the target label's difference
    if desired_y == "P":
        return cur_reach_probs[1] - pre_reach_probs[1]
    else:
        assert desired_y == "N"
        return cur_reach_probs[0] - pre_reach_probs[0]

    # ***Type2: use both target label and opposite label's difference
    # if desired_y == "P":
    #     return (cur_reach_probs[1] - pre_reach_probs[1]) - (cur_reach_probs[0] - pre_reach_probs[0])
    # else:
    #     assert desired_y == "N"
    #     return (cur_reach_probs[0] - pre_reach_probs[0]) - (cur_reach_probs[1] - pre_reach_probs[1])

    # ***Type3: se the target label's difference and add position penalty
    # if desired_y == "P":
    #     return (cur_reach_probs[1] - pre_reach_probs[1]) + 0.1 / (1 + pow(math.e, position))
    # else:
    #     assert desired_y == "N"
    #     return (cur_reach_probs[0] - pre_reach_probs[0]) + 0.1 / (1 + pow(math.e, position))


def sort_words_by_varition(sent, l1_trace, reach_matrix, trans_func):
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
    assert len(l1_trace) - len(sent) == 2, print(len(l1_trace), len(sent))
    c_id = 1  # current state id
    y_predict = l1_trace[-1]
    score_list = []
    reach_probs_list = []
    p = 0
    for sigma, word in zip(l1_trace[1:-1], sent):
        sigma = str(sigma)
        p += 1
        if sigma not in trans_func[c_id]:
            break
            # continue
        else:
            next_id = trans_func[c_id][sigma]
            score = get_word_score_by_significance(y_predict, reach_matrix[c_id - 1], reach_matrix[next_id - 1], p)
            score_list.append(score)
            reach_probs_list.append(reach_matrix[c_id - 1])
            c_id = next_id
    sort_idxs = np.argsort(score_list)[::-1]
    sorted_words = [sent[idx] for idx in sort_idxs]
    sorted_idx = [idx for idx in sort_idxs]
    score_list = [score_list[idx] for idx in sort_idxs]
    return sorted_words, sorted_idx, score_list


def sort_words_by_pprob(sent, l1_trace, trans_func, trans_wfunc):
    '''sort words by drops of path probability.
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
    assert len(l1_trace) - len(sent) == 2, print(len(l1_trace), len(sent))
    score_list = []
    old_prob = get_path_prob([l1_trace], trans_func, trans_wfunc)[0]
    for w_id in range(1, len(l1_trace) - 1):
        new_l1_trace = copy.deepcopy(l1_trace)
        new_l1_trace.pop(w_id)
        new_prob = get_path_prob([new_l1_trace], trans_func, trans_wfunc)[0]
        drop = old_prob - new_prob
        score_list.append(drop)
    sort_idxs = np.argsort(score_list)[::-1]
    sorted_words = [sent[idx] for idx in sort_idxs]
    sorted_idx = [idx for idx in sort_idxs]
    score_list = [score_list[idx] for idx in sort_idxs]
    return sorted_words, sorted_idx, score_list


def sort_words_by_ration(sent, l1_trace, reach_matrix, trans_func):
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
    assert len(l1_trace) - len(sent) == 2, print(len(l1_trace), len(sent))
    c_id = 1  # current state id
    y_predict = l1_trace[-1]
    score_list = []
    for sigma, word in zip(l1_trace[1:-1], sent):
        sigma = str(sigma)
        if sigma not in trans_func[c_id]:
            break
            # continue
        else:
            next_id = trans_func[c_id][sigma]
            reach_probs = reach_matrix[next_id - 1]
            score = get_word_score_by_ration(y_predict, reach_probs)
            score_list.append(score)
            c_id = next_id
    sort_idxs = np.argsort(score_list)[::-1]
    sorted_words = [sent[idx] for idx in sort_idxs]
    sorted_idx = [idx for idx in sort_idxs]
    score_list = [score_list[idx] for idx in sort_idxs]

    return sorted_words, sorted_idx, score_list


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
        # ordered_sent = sort_words_by_rm(sent, l1_trace, reach_matrix, trans_func)
        ordered_sent, sorted_idx, score_list = sort_words_by_varition(sent, l1_trace, reach_matrix, trans_func)
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
