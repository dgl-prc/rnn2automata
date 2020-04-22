"""
predict with the L1 trace.
"""
import sys

sys.path.append("../../../")
from target_models.model_helper import get_model_file, get_input_dim
from experiments.rq4.adv_detect.detect_utils import *
from data.text_utils import is_use_clean


def test_acc_fdlt(test_X, test_Y, model, partitioner, dfa, word2idx, wv_matrix, use_clean=True, input_dim=300,
                  device="cpu", pt_type=PartitionType.KM):
    trans_func, trans_wfunc = dict(dfa["trans_func"]), dict(dfa["trans_wfunc"])
    acc = 0
    fdlt = 0
    unspecified = 0
    for x, y in zip(test_X, test_Y):
        if use_clean:
            x = filter_stop_words(x)
        x_tensor = sent2tensor(x, input_dim, word2idx, wv_matrix, device)
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
            acc += 1
        if pfa_pred == rnn_pred:
            fdlt += 1
        if L2_trace[-1] == "T":
            unspecified += 1
    return acc / len(test_Y), fdlt / len(test_Y), unspecified


if __name__ == '__main__':
    # _data_type = DateSet.IMDB
    # _model_type = ModelType.LSTM
    # _total_symbols = 141953
    ##############################
    # _data_type = DateSet.MR
    # _model_type = ModelType.LSTM
    # _total_symbols = 107355
    ###############################
    # _data_type = DateSet.BP
    # _model_type = ModelType.LSTM
    # _total_symbols = 54044
    ##############################

    for _data_type in [DateSet.Tomita1, DateSet.Tomita2, DateSet.Tomita3, DateSet.Tomita4, DateSet.Tomita5,
                       DateSet.Tomita6]:
        for _model_type in [ModelType.LSTM, ModelType.GRU]:
            print(_data_type.upper(),_model_type.upper())
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

            for _k in range(2, 22, 2):
                pt_path = AbstractData.Level1.NO_STOPW.format(_data_type, _model_type, _k, _data_source + "_partition.pkl")
                _dfa_file_path = AbstractData.Level2.NO_STOPW.format(_data_type, _model_type, _k, alpha)
                _dfa_file_path = os.path.join(_dfa_file_path, "{}_{}_transfunc.pkl").format(_data_source, _total_symbols)
                partitioner = load_pickle(pt_path)
                dfa = load_pickle(get_path(_dfa_file_path))
                acc, fdlt, unspecified = test_acc_fdlt(data["test_x"], data["test_y"], model, partitioner, dfa,
                                                       data["word_to_idx"],
                                                       wv_matrix, use_clean=_use_clean, input_dim=input_dim,
                                                       device=_device, pt_type=_pt_type)
                print(
                    "k={}\tacc={:.4f}\tfdlt={:.4f}\tunspecified:{}/{}".format(_k, acc, fdlt, unspecified, len(data["test_y"])))
