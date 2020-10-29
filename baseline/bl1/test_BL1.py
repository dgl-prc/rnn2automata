import sys
from baseline.bl1.fsa import *
from baseline.baseline_utils import prepare_input
from data.text_utils import is_use_clean, is_artificial
from target_models.model_helper import get_input_dim
from target_models.classifier_adapter import Classifier
from utils.constant import *

if __name__ == '__main__':
    data_groups  = [DateSet.Tomita1, DateSet.Tomita2, DateSet.Tomita3, DateSet.Tomita4, DateSet.Tomita5,
                       DateSet.Tomita6, DateSet.Tomita7]
    k_group = range(2, 12, 2)
    device = "cpu"
    _data_source = "train"
    for _data_type in data_groups:
        for _model_type in [ModelType.LSTM, ModelType.GRU]:
            print(_data_type.upper(), ",", _model_type.upper())
            _use_clean = is_use_clean(_data_type)
            input_dim = get_input_dim(_data_type)
            for _k in k_group:
                l1_traces, wv_matrix, data, model, partitioner = prepare_input(_data_type, _model_type, _data_source,
                                                                               _k)
                classifer = Classifier(model, _model_type, input_dim, data["word_to_idx"], wv_matrix, device)
                ###########
                # build FSA
                ###########
                fsa = FSA(l1_traces, data["train_x"], _k, data["vocab"], model, partitioner, _use_clean)
                #############
                # test FSA
                #############
                acc = 0
                fdlt = 0
                unspecified_cnt = 0
                # test_on = "test" if is_artificial(_data_type) else "train"
                test_on = "test"
                test_size = len(data["{}_y".format(test_on)])
                for x, y in zip(data["{}_x".format(test_on)], data["{}_y".format(test_on)]):
                    if _use_clean:
                        x = filter_stop_words(x)
                    pred, is_unspecified = fsa.predict(x)
                    rnn_pred = classifer.get_label(x)
                    if pred == y:
                        acc += 1
                    if pred == rnn_pred:
                        fdlt += 1
                    if is_unspecified:
                        unspecified_cnt += 1
                acc = acc / test_size
                fdlt = fdlt / test_size
                print(
                    "k={}\tacc={:.4f}\tfdlt={:.4f}\tfinal_states:{}\tunspecified_cnt:{}".format(_k, acc, fdlt,
                                                                                                fsa.final_state,
                                                                                                unspecified_cnt))
