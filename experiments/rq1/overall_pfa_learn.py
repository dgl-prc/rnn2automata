import sys

import shutil

sys.path.append("../../")
from target_models.model_helper import get_model_file, get_input_dim
from experiments.rq4.adv_detect.detect_utils import *
from data.text_utils import is_use_clean
from experiments.rq1.pfa_predict import load_dfa_kits, test_acc_fdlt


def final_output(k, data, model, dfa, total_states, tmp_prims_data, partitioner, pt_type, wv_matrix, use_clean,
                 input_dim, device):
    ###################################
    # show the performance on test data
    ####################################
    test_on = "test_{}"
    acc, fdlt, unspecified = test_acc_fdlt(data[test_on.format('x')], data[test_on.format('y')], model,
                                           partitioner, dfa,
                                           tmp_prims_data,
                                           data["word_to_idx"],
                                           wv_matrix, use_clean=use_clean, input_dim=input_dim,
                                           device=device, pt_type=pt_type)
    print(
        "k={}\t#states={}\tacc={:.4f}\tfdlt={:.4f}\tunspecified:{}/{}".format(k, total_states, acc,
                                                                              fdlt,
                                                                              unspecified,
                                                                              len(data["test_y"])))


#
if __name__ == '__main__':
    _test_on = "train_{}"
    # for _data_type in [DateSet.Tomita1, DateSet.Tomita2, DateSet.Tomita3,
    #                    DateSet.Tomita4, DateSet.Tomita5,
    #                    DateSet.Tomita6, DateSet.Tomita7, DateSet.BP]:
    # _data_type = sys.argv[1]
    _data_type = DateSet.Tomita4
    # _model_type = sys.argv[2]
    # _start_k = int(sys.argv[3])
    # for _model_type in [ModelType.LSTM, ModelType.GRU]:
    for _model_type in [ModelType.GRU]:
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
        #####################
        # model selection
        #####################
        best_dfa = None
        # for _k in range(2, 50, 1):
        for _k in [10]:
            dfa, partitioner, total_states, tmp_prims_data = load_dfa_kits(_data_type, _model_type, _k,
                                                                           _data_source, _total_symbols, alpha)
            if total_states >= 100:
                dfa, partitioner, total_states, tmp_prims_data = load_dfa_kits(_data_type, _model_type, _k - 1,
                                                                               _data_source, _total_symbols, alpha)
                final_output(_k - 1, data, model, dfa, total_states, tmp_prims_data, partitioner, _pt_type,
                             wv_matrix, _use_clean,
                             input_dim, _device)
                shutil.rmtree(tmp_prims_data)
                break
            acc, fdlt, unspecified = test_acc_fdlt(data[_test_on.format('x')], data[_test_on.format('y')], model,
                                                   partitioner, dfa,
                                                   tmp_prims_data,
                                                   data["word_to_idx"],
                                                   wv_matrix, use_clean=_use_clean, input_dim=input_dim,
                                                   device=_device, pt_type=_pt_type)
            if fdlt >= 0.999:
                final_output(_k, data, model, dfa, total_states, tmp_prims_data, partitioner, _pt_type,
                             wv_matrix, _use_clean,
                             input_dim, _device)
                shutil.rmtree(tmp_prims_data)
                break
            print("*******************k={}*****fdlt={}*****************".format(_k, fdlt))
            shutil.rmtree(tmp_prims_data)
