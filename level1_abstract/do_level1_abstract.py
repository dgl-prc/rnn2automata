import sys
sys.path.append("../")
from level1_abstract.clustering_based import *
from utils.help_func import load_pickle
from utils.constant import *
from utils.help_func import save_pickle,load_pickle


def do_L1_abstract(rnn_traces, rnn_pre, k, save_file):
    file_path = os.path.join(get_path(AbstractData.Level1.LSTM.MR), "k={}".format(k), save_file)
    predictor_path = os.path.join(get_path(AbstractData.Level1.LSTM.MR), "k={}".format(k),
                                 save_file.split(".")[0] + "_kmeans.pkl")
    abs_seqs, kmeans = level1_abstract(rnn_traces=rnn_traces, y_pre=rnn_pre, k=k, kmeans_exists=False)
    save_level1_traces(abs_seqs, file_path)
    save_pickle(predictor_path, kmeans)


def do_L1_abstract_on_new_trace(rnn_traces,rnn_pre):
    kmeans = load_pickle("/home/dgl/project/learn_automata_rnn/data/level1_abs_trace/lstm/k=4/test_kmeans.pkl")
    abs_seqs = level1_abstract(rnn_traces=rnn_traces, y_pre=rnn_pre, kmeans=kmeans, kmeans_exists=True)
    print(abs_seqs)




if __name__ == '__main__':
    # for k in range(2, 22, 2):
    #     print("=======k={}======".format(k))
    #     ori_traces = load_pickle(get_path(OriTrace.LSTM.MR))
    #     do_L1_abstract(ori_traces["test_x"], ori_traces["test_pre_y"], k, "test.txt")
    #     do_L1_abstract(ori_traces["train_x"], ori_traces["train_pre_y"], k, "train.txt")
    #     print("done!")
    ori_traces = load_pickle(get_path(OriTrace.LSTM.MR))
    do_L1_abstract_on_new_trace(ori_traces["test_x"], ori_traces["test_pre_y"])
