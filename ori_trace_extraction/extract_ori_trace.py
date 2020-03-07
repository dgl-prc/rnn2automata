from target_models.model_training import sent2tensor
from target_models.model_helper import init_model
from target_models import train_args
from utils.help_func import *
from utils.constant import *


def make_ori_trace(model_type, dataset, device):
    data = load_pickle(get_path(getattr(DataPath, dataset.upper()).PROCESSED_DATA))
    word2idx = data["word_to_idx"]
    wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX))
    input_dim = 300

    params = getattr(train_args, "args_{}_{}".format(model_type, dataset))()
    params["rnn_type"] = model_type
    model = init_model(params=params)

    model_path = get_path(getattr(getattr(TrainedModel, model_type.upper()), dataset.upper()))
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    ori_traces = {}
    ori_traces["train_x"] = []
    ori_traces["test_x"] = []
    ori_traces["train_pre_y"] = []
    ori_traces["test_pre_y"] = []
    print("do extracting...")
    for x in data["train_x"]:
        tensor_sequence = sent2tensor(x, input_dim, word2idx, wv_matrix, device)
        hn_trace, label_trace = model.get_predict_trace(tensor_sequence)
        ori_traces["train_x"].append(hn_trace)
        ori_traces["train_pre_y"].append(label_trace[-1])

    for x in data["test_x"]:
        tensor_sequence = sent2tensor(x, input_dim, word2idx, wv_matrix, device)
        hn_trace, label_trace = model.get_predict_trace(tensor_sequence)
        ori_traces["test_x"].append(hn_trace)
        ori_traces["test_pre_y"].append(label_trace[-1])

    save_path = get_path(getattr(getattr(OriTrace, model_type.upper()), dataset.upper()))
    save_pickle(save_path, ori_traces)
    print("Saved!")
