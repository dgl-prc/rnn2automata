import sys

sys.path.append("../")
from level1_abstract.clustering_based import *
from utils.constant import *
from utils.help_func import save_pickle, load_pickle
from utils.time_util import *
from target_models.model_helper import load_model, get_model_file


def save_L1_result(abs_seqs, partitioner, save_path, data_source):
    if save_path == STANDARD_PATH:
        temp1 = getattr(AbstractData.Level1, partition_type.upper())
        save_folder = get_path(getattr(getattr(temp1, model_type.upper()), data_type.upper()))
    else:
        save_folder = get_path(save_path)
    output_path = save_folder.format(data_type, model_type, k, data_source + ".txt")
    predictor_path = save_folder.format(data_type, model_type, k, data_source + "_partition.pkl")
    save_level1_traces(abs_seqs, output_path)
    save_pickle(predictor_path, partitioner)


def do_L1_abstract(rnn_traces, rnn_pre, k, data_source, model_type, data_type, partition_type, save_path=STANDARD_PATH,
                   model_path=STANDARD_PATH):
    if data_type.startswith("tomita"):
        model = load_model(model_type, "tomita", device="cpu", load_model_path=model_path)
    else:
        model = load_model(model_type, data_type, device="cpu", load_model_path=model_path)

    abs_seqs, partitioner = level1_abstract(rnn=model, rnn_traces=rnn_traces, y_pre=rnn_pre, k=k,
                                            partitioner_exists=False,
                                            partition_type=partition_type)
    save_L1_result(abs_seqs, partitioner, save_path, data_source)


if __name__ == '__main__':
    for data_type in [DateSet.BP, DateSet.Tomita1, DateSet.Tomita2, DateSet.Tomita3,
                       DateSet.Tomita4, DateSet.Tomita5,
                       DateSet.Tomita6, DateSet.Tomita7]:
        for model_type in [ModelType.LSTM, ModelType.GRU]:
            partition_type = PartitionType.KM
            model_file = get_model_file(data_type, model_type)
            _ori_data_path = OriTrace.NO_STOPW.format(data_type, model_type)
            _save_path = AbstractData.Level1.NO_STOPW
            _model_path = TrainedModel.NO_STOPW.format(data_type, model_type, model_file)
            ori_traces = load_pickle(_ori_data_path)
            print("******{}*******{}*******".format(data_type, model_type))
            # for k in range(12, 102, 1):
            for k in range(11, 21, 2):
                print("{}=======k={}======".format(current_timestamp(), k))
                do_L1_abstract(ori_traces["train_x"], ori_traces["train_pre_y"], k, "train", model_type, data_type,
                               partition_type, save_path=_save_path, model_path=_model_path)
            print("{}:done!".format(current_timestamp()))
