from utils.help_func import load_pickle
from utils.constant import *

def topk_acc(k, x, y):
    if len(set(x[:k]) & set(y[:k])) > 0:
        return True
    return 0


def pre_k(k, x, y):
    return len(set(x[:k]) & set(y[:k])) / k


def reciprocal(x, y):
    return 1. / (x.index(y[0]) + 1)


if __name__ == '__main__':
    k = 8
    max_len = 10000
    _data_type = DateSet.IMDB
    _model_type = ModelType.GRU
    data = load_pickle("./sorted_data_{}_{}_k{}.pkl".format(_data_type, _model_type, k))
    for topk in range(1, 6):
        acc = 0
        avg_pre = 0.
        mrr = 0.
        total_size = 0
        for x, y in zip(data["x"], data["y"]):
            if len(x) > max_len:
                continue
            total_size += 1
            if topk_acc(topk, x, y):
                acc += 1
            avg_pre += pre_k(topk, x, y)
            mrr += reciprocal(x, y)
        print("topk={},acc@k:{:.4f}, pre@k:{:.4f},mrr:{:.4f}".format(topk, acc / total_size, avg_pre / total_size,
                                                                     mrr / total_size))
