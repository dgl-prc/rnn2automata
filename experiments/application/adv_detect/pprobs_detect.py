"""
path probability based adversary detection
"""
import sys

sys.path.append("../../../")
from target_models.model_helper import get_model_file
from utils.help_func import get_auc
from experiments.exp_utils import load_dfa, load_partitioner
from experiments.application.adv_detect.detect_utils import *
from experiments.application.adv_detect.textbugger.textbugger_attack import TextBugger


def get_trans_conf(last_inner, trans_wfunc, label):
    pprob = trans_wfunc[last_inner]["P"] if "P" in trans_wfunc[last_inner] else 0.001
    nprob = trans_wfunc[last_inner]["N"] if "N" in trans_wfunc[last_inner] else 0.001
    if label == "P":
        return pprob / nprob
    elif label == "N":
        return nprob / pprob
    else:
        return -1


def do_detect(benign_traces, adv_traces, dfa_file_path=STANDARD_PATH):
    # load dfa
    if dfa_file_path == STANDARD_PATH:
        trans_func, trans_wfunc = load_dfa(_model_type, _data_type, _k, _total_symbols, _data_source, _pt_type)
    else:
        dfa = load_pickle(get_path(dfa_file_path))
        trans_func, trans_wfunc = dict(dfa["trans_func"]), dict(dfa["trans_wfunc"])

    # calculate path prob

    b_scores = []
    for l1_trace in benign_traces:
        benign_prob, b_l2 = get_path_prob(l1_trace, trans_func, trans_wfunc)
        score = get_trans_conf(b_l2[-2], trans_wfunc, l1_trace[-1])
        b_scores.append(score)

    adv_scores = []
    for l1_trace in adv_traces:
        adv_prob, adv_l2 = get_path_prob(l1_trace, trans_func, trans_wfunc)
        score = get_trans_conf(adv_l2[-2], trans_wfunc, l1_trace[-1])
        adv_scores.append(score)

    auc = get_auc(pos_score=b_scores, neg_score=adv_scores)
    # transition undefined
    return auc


if __name__ == '__main__':
    _device = "cpu"
    # _data_type = DateSet.MR
    # _model_type = ModelType.GRU
    # _total_symbols = 141953

    _data_type = DateSet.MR
    _model_type = ModelType.GRU
    _total_symbols = 107355


    _pt_type = PartitionType.KM
    # _data_source = "test"
    _data_source = "train"
    adv_bug_mod = TextBugger.SUB_W
    _use_clean = True
    model_file = get_model_file(_data_type, _model_type)
    _model_path = TrainedModel.NO_STOPW.format(_data_type, _model_type, model_file)
    _adv_path = Application.AEs.NO_STOPW.format(_data_type, _model_type, adv_bug_mod)

    alpha = 64
    for _k in range(2, 22, 2):
        pt_path = AbstractData.Level1.NO_STOPW.format(_data_type, _model_type, _k, _data_source + "_partition.pkl")
        _dfa_file_path = AbstractData.Level2.NO_STOPW.format(_data_type, _model_type, _k, alpha)
        _dfa_file_path = os.path.join(_dfa_file_path, "{}_{}_transfunc.pkl").format(_data_source, _total_symbols)
        #####################
        # load partitioner
        ####################
        if pt_path == STANDARD_PATH:
            partitioner = load_partitioner(_model_type, _data_type, _pt_type, _k, _data_source)
        else:
            partitioner = load_pickle(pt_path)
        benign_abs_seqs, adv_abs_seqs = prepare_L1_data(_model_type, _data_type, _device, partitioner, _pt_type,
                                                        adv_bug_mod, _model_path, _adv_path, _use_clean)
        auc = do_detect(benign_abs_seqs, adv_abs_seqs, _dfa_file_path)
        print("k={}, auc:{:.4f}".format(_k, auc))
