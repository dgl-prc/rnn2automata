"""
path probability based adversary detection
"""
import sys
import shutil

sys.path.append("../../../")
from target_models.model_helper import get_model_file
from utils.help_func import get_auc
from experiments.exp_utils import load_dfa, load_partitioner
from experiments.rq4.adv_detect.detect_utils import *
from experiments.rq4.adv_detect.textbugger.textbugger_attack import TextBugger
from experiments.rq1.get_reachability_matrix import prepare_prism_data, get_state_reachability


def get_trans_conf(last_inner, trans_wfunc, label):
    pprob = trans_wfunc[last_inner]["P"] if "P" in trans_wfunc[last_inner] else 0.001
    nprob = trans_wfunc[last_inner]["N"] if "N" in trans_wfunc[last_inner] else 0.001
    if label == "P":
        return pprob / nprob
    elif label == "N":
        return nprob / pprob
    else:
        return -1


def get_trans_conf_pmc(last_inner, label, pmc_cache, tmp_prims_data):
    if last_inner in pmc_cache:
        probs = pmc_cache[last_inner]
    else:
        probs = get_state_reachability(tmp_prims_data, num_prop=2, start_s=last_inner)
        pmc_cache[last_inner] = probs
    nprob = probs[0] if probs[0] != 0. else 0.0000000000000000000000000000000001
    pprob = probs[1] if probs[1] != 0. else 0.0000000000000000000000000000000001
    if label == "P":
        return pprob / nprob
    elif label == "N":
        return nprob / pprob
    else:
        return -1


def do_detect(benign_traces, adv_traces, trans_func_file, pm_file_path):
    dfa = load_pickle(get_path(trans_func_file))
    trans_func, trans_wfunc = dict(dfa["trans_func"]), dict(dfa["trans_wfunc"])
    _, tmp_prims_data = prepare_prism_data(pm_file_path, num_prop=2)
    pmc_cache = {}
    b_scores = []
    for l1_trace in benign_traces:
        benign_prob, b_l2 = get_path_prob(l1_trace, trans_func, trans_wfunc)
        # score = get_trans_conf(b_l2[-2], trans_wfunc, l1_trace[-1])
        score = get_trans_conf_pmc(b_l2[-2], l1_trace[-1], pmc_cache, tmp_prims_data)
        b_scores.append(score)

    adv_scores = []
    for l1_trace in adv_traces:
        adv_prob, adv_l2 = get_path_prob(l1_trace, trans_func, trans_wfunc)
        # score = get_trans_conf(adv_l2[-2], trans_wfunc, l1_trace[-1])
        score = get_trans_conf_pmc(adv_l2[-2], l1_trace[-1], pmc_cache, tmp_prims_data)
        adv_scores.append(score)
    shutil.rmtree(tmp_prims_data)
    auc = get_auc(pos_score=b_scores, neg_score=adv_scores)
    return auc


if __name__ == '__main__':
    _device = "cpu"
    _data_type = sys.argv[1]
    _model_type = sys.argv[2]
    k = int(sys.argv[3])
    _total_symbols = get_total_symbols(_data_type)

    _pt_type = PartitionType.KM
    # _data_source = "test"
    _data_source = "train"
    adv_bug_mod = TextBugger.SUB_W
    _use_clean = True
    model_file = get_model_file(_data_type, _model_type)
    _model_path = TrainedModel.NO_STOPW.format(_data_type, _model_type, model_file)
    _adv_path = Application.AEs.NO_STOPW.format(_data_type, _model_type, adv_bug_mod)

    alpha = 64
    for _k in [k]:
        pt_path = AbstractData.Level1.NO_STOPW.format(_data_type, _model_type, _k, _data_source + "_partition.pkl")
        _dfa_file_path = AbstractData.Level2.NO_STOPW.format(_data_type, _model_type, _k, alpha)
        trans_func_file = os.path.join(_dfa_file_path, "{}_{}_transfunc.pkl").format(_data_source, _total_symbols)
        pm_file_path = os.path.join(_dfa_file_path, "{}_{}.pm").format(_data_source,
                                                                       _total_symbols)
        partitioner = load_pickle(pt_path)
        benign_abs_seqs, adv_abs_seqs = prepare_L1_data(_model_type, _data_type, _device, partitioner, _pt_type,
                                                        adv_bug_mod, _model_path, _adv_path, _use_clean)
        auc = do_detect(benign_abs_seqs, adv_abs_seqs, trans_func_file, pm_file_path)
        print("k={}, auc:{:.4f}".format(_k, auc))
