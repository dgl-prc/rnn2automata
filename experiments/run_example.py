import sys

sys.path.append("../../")
from target_models.model_helper import get_model_file, get_input_dim
from experiments.application.adv_detect.detect_utils import *
from data.text_utils import is_use_clean
from experiments.effectiveness.get_reachability_matrix import prepare_prism_data, get_state_reachability


def show_runexamples(test_X, test_Y, model, partitioner, dfa, word2idx, wv_matrix, use_clean=True, input_dim=300,
                     device="cpu", pt_type=PartitionType.KM):
    trans_func, trans_wfunc = dict(dfa["trans_func"]), dict(dfa["trans_wfunc"])
    for ori_x, y in zip(test_X, test_Y):
        if use_clean:
            pure_x = filter_stop_words(ori_x)
        x_tensor = sent2tensor(pure_x, input_dim, word2idx, wv_matrix, device)
        hn_trace, label_trace = model.get_predict_trace(x_tensor)
        rnn_pred = label_trace[-1]
        L1_trace = level1_abstract(rnn=None, rnn_traces=[hn_trace], y_pre=[rnn_pred],
                                   partitioner=partitioner,
                                   partitioner_exists=True, partition_type=pt_type)[0]
        _, L2_trace = get_path_prob(L1_trace, trans_func, trans_wfunc)
        last_inner = L2_trace[-2]
        pprob = trans_wfunc[last_inner]["P"] if "P" in trans_wfunc[last_inner] else 0.
        nprob = trans_wfunc[last_inner]["N"] if "N" in trans_wfunc[last_inner] else 0.
        pfa_pred = np.argmax([nprob, pprob])

        if pfa_pred == y:
            print("=======================================")
            print("ORI_TRACE:{}".format(" ".join(ori_x)))
            print("PURE_TRACE:{}".format(" ".join(pure_x)))
            print("LABEL:{}".format(y))
            print("L1_TRACE:{}".format(L1_trace))
            print("concrete trace:")
            print(["{:.4f}".format(v) for v in [hn_trace[0][0], hn_trace[0][1], hn_trace[0][-1]]])
            print(["{:.4f}".format(v) for v in [hn_trace[-1][0], hn_trace[-1][1], hn_trace[-1][-1]]])

def find_example():
    _data_type = DateSet.MR
    _model_type = ModelType.GRU
    _k = 2
    _total_symbols = get_total_symbols(_data_type)
    print(_data_type.upper(), _model_type.upper())
    _total_symbols = get_total_symbols(_data_type)
    _device = "cpu"
    _pt_type = PartitionType.KM
    _data_source = "train"
    _use_clean = is_use_clean(_data_type)
    alpha = 64
    input_dim = get_input_dim(_data_type)
    model_file = get_model_file(_data_type, _model_type)
    _model_path = TrainedModel.NO_STOPW.format(_data_type, _model_type, model_file)

    model = load_model(_model_type, _data_type, device="cpu", load_model_path=_model_path)
    data = load_pickle(get_path(getattr(DataPath, _data_type.upper()).PROCESSED_DATA))
    wv_matrix = load_pickle(get_path(getattr(DataPath, _data_type.upper()).WV_MATRIX))

    pt_path = AbstractData.Level1.NO_STOPW.format(_data_type, _model_type, _k, _data_source + "_partition.pkl")
    _dfa_file_path = AbstractData.Level2.NO_STOPW.format(_data_type, _model_type, _k, alpha)
    _dfa_file_path = os.path.join(_dfa_file_path, "{}_{}_transfunc.pkl").format(_data_source, _total_symbols)
    partitioner = load_pickle(pt_path)
    dfa = load_pickle(get_path(_dfa_file_path))
    show_runexamples(data["test_x"], data["test_y"], model, partitioner, dfa,
                     data["word_to_idx"],
                     wv_matrix, use_clean=_use_clean, input_dim=input_dim,
                     device=_device, pt_type=_pt_type)


def get_reachability():
    pm_file_path = "/home/dgl/project/learn_automata_rnn/data/no_stopws/L2_results/mr/gru/k=2/alpha=64/train_107355.pm"
    total_states, tmp_prims_data = prepare_prism_data(pm_file_path, num_prop=2)
    probs = get_state_reachability(tmp_prims_data, num_prop=2, start_s='2')
    print(probs)


if __name__ == '__main__':
    get_reachability()