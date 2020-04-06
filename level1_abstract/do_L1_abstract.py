import sys

sys.path.append("../")
from level1_abstract.clustering_based import *
from utils.constant import *
from utils.help_func import save_pickle, load_pickle
from utils.time_util import *
from target_models.model_helper import load_model, get_model_file


def do_L1_abstract(rnn_traces, rnn_pre, k, data_source, model_type, data_type, partition_type, save_path=STANDARD_PATH,
                   model_path=STANDARD_PATH):
    if save_path == STANDARD_PATH:
        temp1 = getattr(AbstractData.Level1, partition_type.upper())
        save_folder = get_path(getattr(getattr(temp1, model_type.upper()), data_type.upper()))
    else:
        save_folder = get_path(save_path)

    output_path = save_folder.format(data_type, model_type, k, data_source + ".txt")
    predictor_path = save_folder.format(data_type, model_type, k, data_source + "_partition.pkl")
    rnn = load_model(model_type, data_type, device="cpu", load_model_path=model_path)
    abs_seqs, partitioner = level1_abstract(rnn=rnn, rnn_traces=rnn_traces, y_pre=rnn_pre, k=k,
                                            partitioner_exists=False,
                                            partition_type=partition_type)
    save_level1_traces(abs_seqs, output_path)
    save_pickle(predictor_path, partitioner)


if __name__ == '__main__':
    data_type = sys.argv[1]
    model_type = sys.argv[2]
    partition_type = PartitionType.KM
    model_file = get_model_file(data_type, model_type)
    _ori_data_path = OriTrace.NO_STOPW.format(data_type, model_type)
    _save_path = AbstractData.Level1.NO_STOPW
    _model_path = TrainedModel.NO_STOPW.format(data_type, model_type, model_file)
    if _ori_data_path == STANDARD_PATH:
        _ori_data_path = get_path(OriTrace.LSTM.MR)
    ori_traces = load_pickle(_ori_data_path)
    print("******{}*******{}*******".format(data_type, model_type))
    for k in range(2, 22, 2):
        # k = int(sys.argv[1])
        print("{}=======k={}======".format(current_timestamp(), k))
        do_L1_abstract(ori_traces["train_x"], ori_traces["train_pre_y"], k, "train", model_type, data_type,
                       partition_type, save_path=_save_path, model_path=_model_path)
        do_L1_abstract(ori_traces["test_x"], ori_traces["test_pre_y"], k, "test", model_type, data_type,
                       partition_type, save_path=_save_path, model_path=_model_path)
    print("{}:done!".format(current_timestamp()))
