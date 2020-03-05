from level1_abstract.clustering_based import *
from utils.help_func import load_pickle
from utils.constant import *


def do_L1_abstract(rnn_traces, rnn_pre, k, save_file):
    abs_seqs = level1_abstract(rnn_traces, rnn_pre, k)
    file_path = os.path.join(get_path(AbstractData.Level1.LSTM.MR), "k={}".format(k), save_file)
    save_level1_traces(abs_seqs, file_path)


if __name__ == '__main__':
    for k in range(2, 20, 2):
        print("=======k={}======".format(k))
        ori_traces = load_pickle(get_path(OriTrace.LSTM.MR))
        do_L1_abstract(ori_traces["test_x"], ori_traces["test_pre_y"], k, "test.txt")
        do_L1_abstract(ori_traces["train_x"], ori_traces["train_pre_y"], k, "train.txt")
        print("done!")
