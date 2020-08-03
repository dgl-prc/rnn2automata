"""
predict with the L1 trace.
"""
import sys

import shutil

sys.path.append("../../")
from target_models.model_helper import get_model_file, get_input_dim
from experiments.rq4.adv_detect.detect_utils import *
from data.text_utils import is_use_clean
from experiments.rq1.get_reachability_matrix import prepare_prism_data, get_state_reachability


def load_dfa_kits(data_type, model_type, k, data_source, total_symbols, alpha):
    pt_path = AbstractData.Level1.NO_STOPW.format(data_type, model_type, k,
                                                  data_source + "_partition.pkl")
    dfa_file_path = AbstractData.Level2.NO_STOPW.format(data_type, model_type, k, alpha)
    trans_func_file = os.path.join(dfa_file_path, "{}_{}_transfunc.pkl").format(data_source,
                                                                                total_symbols)
    pm_file_path = os.path.join(dfa_file_path, "{}_{}.pm").format(data_source,
                                                                  total_symbols)
    partitioner = load_pickle(pt_path)
    dfa = load_pickle(get_path(trans_func_file))
    # make reachability matrix
    total_states, tmp_prims_data = prepare_prism_data(pm_file_path, num_prop=2)
    return dfa, partitioner, total_states, tmp_prims_data


def extract_l1_trace(x, model, input_dim, word2idx, wv_matrix, device, partitioner, pt_type, use_clean):
    if use_clean:
        x = filter_stop_words(x)
    x_tensor = sent2tensor(x, input_dim, word2idx, wv_matrix, device)
    hn_trace, label_trace = model.get_predict_trace(x_tensor)
    rnn_pred = label_trace[-1]
    L1_trace = level1_abstract(rnn=None, rnn_traces=[hn_trace], y_pre=[rnn_pred],
                               partitioner=partitioner,
                               partitioner_exists=True, partition_type=pt_type)[0]
    return L1_trace


def test_acc_fdlt(**kwargs):
    test_X = kwargs["X"]
    test_Y = kwargs["Y"]
    dfa = kwargs["dfa"]
    tmp_prims_data = kwargs["tmp_prims_data"]
    if kwargs["input_type"] == "text":
        model = kwargs["model"]
        partitioner = kwargs["partitioner"]
        word2idx = kwargs["word2idx"]
        wv_matrix = kwargs["wv_matrix"]
        use_clean = kwargs["use_clean"]
        input_dim = kwargs["input_dim"]
        device = kwargs["device"]
        pt_type = kwargs["pt_type"]
        is_text = True
    else:
        is_text = False
    trans_func, trans_wfunc = dict(dfa["trans_func"]), dict(dfa["trans_wfunc"])
    acc = 0
    fdlt = 0
    unspecified = 0
    pmc_cache = {}
    for x, y in zip(test_X, test_Y):
        if is_text:
            L1_trace = extract_l1_trace(x, model, input_dim, word2idx, wv_matrix, device, partitioner, pt_type,
                                        use_clean)
        else:
            L1_trace = x
        rnn_pred = 0 if L1_trace[-1] == 'N' else 1
        _, L2_trace = get_path_prob(L1_trace, trans_func, trans_wfunc)
        last_inner = L2_trace[-2]
        if last_inner in pmc_cache:
            probs = pmc_cache[last_inner]
        else:
            probs = get_state_reachability(tmp_prims_data, num_prop=2, start_s=last_inner)
            pmc_cache[last_inner] = probs
        pfa_pred = np.argmax(probs)
        if pfa_pred == y:
            acc += 1
        if pfa_pred == rnn_pred:
            fdlt += 1
        if L2_trace[-1] == "T":
            unspecified += 1
    return acc / len(test_Y), fdlt / len(test_Y), unspecified


if __name__ == '__main__':
    test_on = "test_{}"
    for _data_type in [DateSet.BP, DateSet.Tomita1, DateSet.Tomita2, DateSet.Tomita3,
                       DateSet.Tomita4, DateSet.Tomita5, DateSet.Tomita6, DateSet.Tomita7,
                       DateSet.MR, DateSet.IMDB]:
        for _model_type in [ModelType.LSTM, ModelType.GRU]:
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

            ###########################
            # load model and test data
            ###########################
            if _data_type.startswith("tomita"):
                gram_id = int(_data_type[-1])
                data = load_pickle(get_path(getattr(DataPath, "TOMITA").PROCESSED_DATA).format(gram_id, gram_id))
                wv_matrix = load_pickle(get_path(getattr(DataPath, "TOMITA").WV_MATRIX).format(gram_id, gram_id))
                model = load_model(_model_type, "tomita", device="cpu", load_model_path=_model_path)
            else:
                model = load_model(_model_type, _data_type, device="cpu", load_model_path=_model_path)
                data = load_pickle(get_path(getattr(DataPath, _data_type.upper()).PROCESSED_DATA))
                wv_matrix = load_pickle(get_path(getattr(DataPath, _data_type.upper()).WV_MATRIX))

            # for _k in range(61, 66, 1):
            # for _k in range(2, 12, 2):
            for _k in [6]:
                pt_path = AbstractData.Level1.NO_STOPW.format(_data_type, _model_type, _k,
                                                              _data_source + "_partition.pkl")
                dfa_file_path = AbstractData.Level2.NO_STOPW.format(_data_type, _model_type, _k, alpha)
                trans_func_file = os.path.join(dfa_file_path, "{}_{}_transfunc.pkl").format(_data_source,
                                                                                            _total_symbols)
                pm_file_path = os.path.join(dfa_file_path, "{}_{}.pm").format(_data_source,
                                                                              _total_symbols)
                partitioner = load_pickle(pt_path)
                dfa = load_pickle(get_path(trans_func_file))
                # make reachability matrix
                total_states, tmp_prims_data = prepare_prism_data(pm_file_path, num_prop=2)
                acc, fdlt, unspecified = test_acc_fdlt(data[test_on.format('x')], data[test_on.format('y')], model,
                                                       partitioner, dfa,
                                                       tmp_prims_data,
                                                       data["word_to_idx"],
                                                       wv_matrix, use_clean=_use_clean, input_dim=input_dim,
                                                       device=_device, pt_type=_pt_type)
                print(
                    "k={}\t#states={}\tacc={:.4f}\tfdlt={:.4f}\tunspecified:{}/{}".format(_k, total_states, acc, fdlt,
                                                                                          unspecified,
                                                                                          len(data["test_y"])))
                shutil.rmtree(tmp_prims_data)
