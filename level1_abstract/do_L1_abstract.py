import sys

sys.path.append("../")
from level1_abstract.clustering_based import *
from utils.constant import *
from utils.help_func import save_pickle, load_pickle
from utils.time_util import *


def do_L1_abstract(rnn_traces, rnn_pre, k, save_file, model_type, data_type, partition_type):
    temp1 = getattr(AbstractData.Level1, partition_type.upper())
    save_folder = get_path(getattr(getattr(temp1, model_type.upper()), data_type.upper()))
    output_path = os.path.join(save_folder, "k={}".format(k), save_file)
    predictor_path = os.path.join(save_folder, "k={}".format(k),
                                  save_file.split(".")[0] + "_partition.pkl")
    abs_seqs, partitioner = level1_abstract(rnn_traces=rnn_traces, y_pre=rnn_pre, k=k, partitioner_exists=False,
                                            partition_type="hc")
    save_level1_traces(abs_seqs, output_path)
    save_pickle(predictor_path, partitioner)


if __name__ == '__main__':
    partition_type = "hc"
    model_type = ModelType.LSTM
    data_type = DateSet.MR
    for k in range(8, 22, 2):
    # k = int(sys.argv[1])
        print("{}=======k={}======".format(current_timestamp(), k))
        ori_traces = load_pickle(get_path(OriTrace.LSTM.MR))
        do_L1_abstract(ori_traces["test_x"], ori_traces["test_pre_y"], k, "test.txt", model_type, data_type,
                       partition_type)
        # do_L1_abstract(ori_traces["train_x"], ori_traces["train_pre_y"], k, "train.txt", model_type, data_type,
        #                partition_type)
    print("{}:done!".format(current_timestamp()))
