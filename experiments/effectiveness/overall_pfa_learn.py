import sys

import shutil
import time

sys.path.append("../../")
from target_models.model_helper import get_model_file, get_input_dim
from experiments.rq4.adv_detect.detect_utils import *
from data.text_utils import is_use_clean
from experiments.rq1.pfa_predict import load_dfa_kits, test_acc_fdlt
from level2_abstract.aalergia import *
from experiments.rq1.get_reachability_matrix import prepare_prism_data


def final_output(k, dfa, total_states, tmp_prims_data, partitioner, pt_type, use_clean,
                 input_dim, device):
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
    ###################################
    # show the performance on test data
    ####################################
    test_x = data["test_x"]
    test_y = data["test_y"]
    acc, fdlt, unspecified = test_acc_fdlt(X=test_x, Y=test_y, model=model,
                                           partitioner=partitioner, dfa=dfa,
                                           tmp_prims_data=tmp_prims_data,
                                           word2idx=data["word_to_idx"],
                                           wv_matrix=wv_matrix, use_clean=use_clean, input_dim=input_dim,
                                           device=device, pt_type=pt_type, input_type="text")
    print(
        "k={}\t#states={}\tacc={:.4f}\tfdlt={:.4f}\tunspecified:{}/{}".format(k, total_states, acc,
                                                                              fdlt,
                                                                              unspecified,
                                                                              len(data["test_y"])))


def overall(output_path, alpha, ori_traces, gamma_a, time_out):
    start_time = time.clock()
    k = 1
    lst_f = 0.0
    is_timeout = True
    while time.clock() - start_time <= time_out:
        k += 1
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        abs_seqs, partitioner = level1_abstract(rnn=None, rnn_traces=ori_traces["train_x"],
                                                y_pre=ori_traces["train_pre_y"], k=k,
                                                partitioner_exists=False,
                                                partition_type=PartitionType.KM)
        alphabet = set(["S", "P", "N"] + [str(i) for i in range(k)])
        al = AALERGIA(alpha, abs_seqs, alphabet, start_symbol=START_SYMBOL, output_path=output_path,
                      show_merge_info=False)
        dffa = al.learn()
        pfa, pm_path = al.output_prism(dffa, model_name="{}_{}_{}".format(_data_type, _model_type, k))
        total_states, tmp_prims_data = prepare_prism_data(pm_path, num_prop=2)
        _, fdlt, unspecified = test_acc_fdlt(X=abs_seqs, Y=ori_traces["train_pre_y"],
                                             dfa=pfa, tmp_prims_data=tmp_prims_data,
                                             input_type="L1")
        print(">>>>>fdlt:{:.4f}".format(fdlt))
        if fdlt >= gamma_a:
            break
        if fdlt < lst_f:
            is_timeout = False
            break
        else:
            lst_f = fdlt
            last_pfa = pfa
            last_total_states = total_states
            last_tmp_prims_data = tmp_prims_data
            last_partitioner = partitioner
            last_k = k
    if is_timeout:
        return pfa, total_states, tmp_prims_data, partitioner, k
    else:
        return last_pfa, last_total_states, last_tmp_prims_data, last_partitioner, last_k


if __name__ == '__main__':
    _device = "cpu"
    _alpha = 64
    time_out = int(sys.argv[1])
    # time_out = 400
    # time_out = 1200
    for _data_type in [DateSet.Tomita1, DateSet.Tomita2, DateSet.Tomita3,
                       DateSet.Tomita4, DateSet.Tomita5, DateSet.Tomita6,
                       DateSet.Tomita7, DateSet.BP, DateSet.MR, DateSet.IMDB]:
    # for _data_type in [DateSet.MR, DateSet.IMDB]:
        for _model_type in [ModelType.LSTM, ModelType.GRU]:
            print("=============={}==============={}==============".format(_data_type.upper(),
                                                                           _model_type.upper()))
            _total_symbols = get_total_symbols(_data_type)
            input_dim = get_input_dim(_data_type)
            use_clean = is_use_clean(_data_type)
            _pt_type = PartitionType.KM
            _data_source = "train"
            _use_clean = is_use_clean(_data_type)
            input_dim = get_input_dim(_data_type)
            model_file = get_model_file(_data_type, _model_type)
            _model_path = TrainedModel.NO_STOPW.format(_data_type, _model_type, model_file)
            _ori_data_path = OriTrace.NO_STOPW.format(_data_type, _model_type)
            _ori_traces = load_pickle(_ori_data_path)
            ##############
            # overall learn
            ###############
            _output_path = "./tmp/{}/{}/".format(_data_type, _model_type)
            _gamma_a = 0.99
            pfa, total_states, tmp_prims_data, partitioner, k = overall(_output_path, _alpha, _ori_traces, _gamma_a,
                                                                        time_out)
            ###########
            # test
            ##########
            final_output(k, pfa, total_states, tmp_prims_data, partitioner, pt_type=_pt_type, use_clean=use_clean,
                         input_dim=input_dim, device=_device)
            shutil.rmtree(tmp_prims_data)
            shutil.rmtree(_output_path)
