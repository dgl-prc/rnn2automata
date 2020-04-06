import random
from utils.constant import *
from utils.help_func import load_pickle


def select_benign_data(classifier, data):
    acc = 0
    benigns = []
    idx = []
    i = 0
    for sent, label in zip(data["test_x"], data["test_y"]):
        pred = classifier.get_label(sent)
        if pred == label:
            acc += 1
            benigns.append((sent, label))
            idx.append(i)
        i += 1
    random.seed(20200306)
    rnd_dicies = [i for i in range(len(idx))]
    random.shuffle(rnd_dicies)
    print("Acc:{:.4f}".format(acc / len(data["test_x"])))
    return [benigns[i] for i in rnd_dicies], [idx[i] for i in rnd_dicies]


# def load_dfa(model_type, data_type, k, total_symbols, data_source, pt_type):
#     '''
#     Parameters.
#     ------------
#     model_type:
#     data_type:
#     k:
#     total_symbols:
#     data_source: built on test set or train set
#     pt_type: partition type
#     :return:
#     '''
#     l2_path = getattr(getattr(getattr(AbstractData.Level2, pt_type.upper()), model_type.upper()), data_type.upper())
#     tranfunc_path = get_path(
#         os.path.join(l2_path, data_source,
#                      "{}_{}_k{}_{}_transfunc.pkl".format(model_type, data_type, k, total_symbols)))
#     dfa = load_pickle(tranfunc_path)
#     trans_func, trans_wfunc = dict(dfa["trans_func"]), dict(dfa["trans_wfunc"])
#     return trans_func, trans_wfunc

# just for debug
def load_dfa(model_type, data_type, k, total_symbols, data_source, pt_type, alpha=None):
    '''
    Parameters.
    ------------
    model_type:
    data_type:
    k:
    total_symbols:
    data_source: built on test set or train set
    pt_type: partition type
    :return:
    '''
    l2_path = getattr(getattr(getattr(AbstractData.Level2, pt_type.upper()), model_type.upper()), data_type.upper())
    if alpha:
        tranfunc_path = get_path(
            os.path.join(l2_path, data_source,
                         "{}_{}_k{}_alpha-{}_{}_transfunc.pkl".format(model_type, data_type, k, alpha, total_symbols)))
    else:
        tranfunc_path = get_path(
            os.path.join(l2_path, data_source,
                         "{}_{}_k{}_{}_transfunc.pkl".format(model_type, data_type, k, total_symbols)))

    dfa = load_pickle(tranfunc_path)
    trans_func, trans_wfunc = dict(dfa["trans_func"]), dict(dfa["trans_wfunc"])
    return trans_func, trans_wfunc


def load_partitioner(model_type, data_type, pt_type, k, data_source):
    L1_abs_folder = getattr(AbstractData.Level1, pt_type.upper())
    L1_abs_folder = getattr(L1_abs_folder, model_type.upper())
    L1_abs_folder = getattr(L1_abs_folder, data_type.upper())
    if pt_type == "km":
        # Legacy issues
        # cluster_path = os.path.join(L1_abs_folder, "k={}".format(k), "{}_kmeans.pkl".format(data_source))
        cluster_path = os.path.join(L1_abs_folder, "k={}".format(k), "{}_partition.pkl".format(data_source))
    else:
        cluster_path = os.path.join(L1_abs_folder, "k={}".format(k), "{}_partition.pkl".format(data_source))

    partitioner = load_pickle(get_path(cluster_path))
    return partitioner
