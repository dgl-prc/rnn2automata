import sys
import copy
import rbo

sys.path.append("../../../")
from experiments.application.find_important_words.sort_words import *
from target_models.model_helper import load_model, get_model_file
from target_models.classifier_adapter import Classifier
from data.text_utils import filter_stop_words
from utils.help_func import save_pickle


def evaluate_by_confidence(sent, sorted_idx, classifier, topk):
    avg_down = []
    y_pre = classifier.get_label(sent)
    for target_idx in sorted_idx[:topk]:
        new_sent = copy.deepcopy(sent)
        new_sent.pop(target_idx)
        old_confidence = classifier.get_probs(sent)[y_pre]
        new_confidence = classifier.get_probs(new_sent)[y_pre]
        # assume that without the target word, the confidence of the target label will decline
        drop = old_confidence - new_confidence
        avg_down.append(drop)
    return np.average(avg_down)


def evaluate_by_confidence_rbo(sent, reacha_sorted_idx, classifier, topk):
    y_pre = classifier.get_label(sent)
    new_reach_sorted = []
    confidence_drop = []
    for target_idx in reacha_sorted_idx:
        # # filter stop words
        # if sent[target_idx] in stop_words:
        #     continue
        new_reach_sorted.append(target_idx)
        new_sent = copy.deepcopy(sent)
        new_sent.pop(target_idx)
        old_confidence = classifier.get_probs(sent)[y_pre]
        new_confidence = classifier.get_probs(new_sent)[y_pre]
        # assume that without the target word, the confidence of the target label will decline
        drop = old_confidence - new_confidence
        confidence_drop.append(drop)
    temp_idxs = np.argsort(confidence_drop)[::-1]
    confidence_sorted = [new_reach_sorted[idx] for idx in temp_idxs]
    rbo_score = rbo.RankingSimilarity(new_reach_sorted[:topk], confidence_sorted[:topk]).rbo()
    print("*************************Y-PRE: {}************************************".format(y_pre))
    predicted_list = [sent[idx] for idx in new_reach_sorted]
    grnd_list = [sent[idx] for idx in confidence_sorted]
    print("Ori:{}".format(" ".join(sent)))
    print("Reach-Based:{}".format(predicted_list))
    print("Conf-Based:{}".format(grnd_list))
    print("Conf-Drops:{}".format([confidence_drop[idx] for idx in temp_idxs]))
    print("RBO:{}".format(rbo_score))
    return rbo_score, predicted_list, grnd_list


def main(**kwargs):
    model_type = kwargs["m_tp"]
    data_type = kwargs["d_tp"]
    data_source = kwargs["d_s"]
    pt_type = kwargs["pt_tp"]
    total_symbols = kwargs["t_syms"]
    k = kwargs["k"]
    num_prop = kwargs["num_prop"]
    top_ration = kwargs["top_ration"]
    min_len = kwargs["min_len"]
    max_len = kwargs["max_len"]
    test_on_train = kwargs["test_on_train"]
    max_samples = kwargs["max_samples"]
    omit_stopws = kwargs["omit_stopws"]
    pm_file_path = kwargs["pm_file_path"]
    dfa_file_path = kwargs["dfa_file_path"]
    model_path = kwargs["model_path"]
    partitioner_path = kwargs["partitioner_path"]

    #############
    # load data
    ##############
    if pm_file_path == STANDARD_PATH:
        pm_file = getattr(getattr(getattr(AbstractData.Level2, pt_type.upper()), model_type.upper()), data_type.upper())
        pm_file = os.path.join(get_path(pm_file), data_source,
                               "{}_{}_k{}_{}.pm".format(model_type, data_type, k, total_symbols))
    else:
        pm_file = get_path(pm_file_path)

    if dfa_file_path == STANDARD_PATH:
        trans_func, trans_wfunc = load_dfa(model_type, data_type, k, total_symbols, data_source, pt_type)
    else:
        dfa = load_pickle(get_path(dfa_file_path))
        trans_func, trans_wfunc = dict(dfa["trans_func"]), dict(dfa["trans_wfunc"])

    #####################
    # load partitioner
    if partitioner_path == STANDARD_PATH:
        partitioner = load_partitioner(model_type, data_type, pt_type, k, data_source)
    else:
        partitioner = load_pickle(partitioner_path)

    reach_matrix = reachability_matrix(pm_file, num_prop)
    raw_data = load_pickle(get_path(getattr(DataPath, data_type.upper()).PROCESSED_DATA))
    wv_matrix = load_pickle(get_path(getattr(DataPath, data_type.upper()).WV_MATRIX))
    word2idx = raw_data["word_to_idx"]

    ############
    # load model
    ############
    device = "cpu"
    model = load_model(model_type, data_type, device, load_model_path=model_path)
    classifier = Classifier(model, model_type, 300, raw_data["word_to_idx"], wv_matrix, device)

    cnt1 = 0
    cnt2 = 0
    num_empty = 0
    num_error = 0
    avg_rbo = []  # mean average down
    cnt = 0
    if test_on_train:
        test_X, test_Y = raw_data["train_x"], raw_data["train_y"]
    else:
        test_X, test_Y = raw_data["test_x"], raw_data["test_y"]

    sorted_rst = {"x": [], "y": []}
    for sent, y_true in zip(test_X, test_Y):
        if cnt > max_samples:
            break
        # only test on the benign data
        if y_true != classifier.get_label(sent):
            continue
        if omit_stopws:
            sent = filter_stop_words(sent)
        if len(sent) > max_len or len(sent) < min_len:
            continue
        cnt += 1
        l1_trace = sent2L1_trace(sent, model, word2idx, wv_matrix, device, partitioner, pt_type)
        topk = int(top_ration * len(sent))
        ordered_words, sorted_idx, score_list = sort_words_by_varition(sent, l1_trace, reach_matrix, trans_func)
        # ordered_words, sorted_idx, score_list = sort_words_by_pprob(sent, l1_trace, trans_func, trans_wfunc)
        rbo_score, predicted_list, grnd_list = evaluate_by_confidence_rbo(sent, sorted_idx, classifier, topk)
        sorted_rst["x"].append(predicted_list)
        sorted_rst["y"].append(grnd_list)
        avg_rbo.append(rbo_score)
        if len(ordered_words) == 0:
            num_empty += 1
            if (y_true == 1 and l1_trace[-1] == "N") or (y_true == 0 and l1_trace[-1] == "P"):
                cnt1 += 1

        if (y_true == 1 and l1_trace[-1] == "N") or (y_true == 0 and l1_trace[-1] == "P"):
            num_error += 1
            if len(ordered_words) == 0:
                cnt2 += 1
        # sys.stdout.write("\rprocessing...{:.2f}%".format(cnt*100/len(raw_data["test_x"])))
        # sys.stdout.flush()
    avg_rbo = np.average(avg_rbo)
    print("\navg_rbo: {}".format(avg_rbo))
    print("Total test samples:{}".format(cnt))
    save_pickle("./sorted_data_{}_{}_k{}.pkl".format(data_type, model_type, k), sorted_rst)
    # save_pickle("./sorted_data_k{}_pp-limit.pkl".format(k), sorted_rst)


if __name__ == '__main__':
    _num_prop = 2
    _k = 8
    _alpha = 64
    _data_type = DateSet.IMDB
    _model_type = ModelType.GRU
    _data_source = "train"
    # _total_symbols = 196425
    _total_symbols = 141953
    # _total_symbols = 107563
    # _data_source = "test"
    # _total_symbols = 49056
    _pt_type = PartitionType.KM
    _top_ration = 1
    _max_len_sent = 10000
    _min_len_sent = 3
    _test_on_train = False
    _omit_stopws = True
    _max_samples = 1000

    model_file = get_model_file(_data_type, _model_type)
    _model_path = TrainedModel.NO_STOPW.format(_data_type, _model_type, model_file)
    pt_path = AbstractData.Level1.NO_STOPW.format(_data_type, _model_type, _k, _data_source + "_partition.pkl")
    _dfa_file_path = AbstractData.Level2.NO_STOPW.format(_data_type, _model_type, _k, _alpha)
    _dfa_file_path = os.path.join(_dfa_file_path, "{}_{}_transfunc.pkl").format(_data_source, _total_symbols)
    _pm_file_path = AbstractData.Level2.NO_STOPW.format(_data_type, _model_type, _k, _alpha)
    _pm_file_path = os.path.join(_pm_file_path, "{}_{}.pm").format(_data_source, _total_symbols)

    min_compred = int(_min_len_sent * _top_ration)
    print("**********minimum sentence length:{}, compared ration:{}***********".format(_min_len_sent,
                                                                                       _top_ration))
    main(m_tp=_model_type, d_tp=_data_type, d_s=_data_source, pt_tp=_pt_type, t_syms=_total_symbols, k=_k,
         num_prop=_num_prop, top_ration=_top_ration, min_len=_min_len_sent, max_len=_max_len_sent,
         test_on_train=_test_on_train, max_samples=_max_samples, omit_stopws=_omit_stopws, pm_file_path=_pm_file_path,
         dfa_file_path=_dfa_file_path, model_path=_model_path, partitioner_path=pt_path)
