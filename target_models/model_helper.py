import torch
from target_models import train_args
from target_models.gated_rnn import *


def init_model(params):
    if params["rnn_type"] == ModelType.GRU:
        model = GRU(input_size=params["input_size"], num_class=params["output_size"], hidden_size=params["hidden_size"],
                    num_layers=params["num_layers"])
    elif params["rnn_type"] == ModelType.LSTM:
        model = LSTM(input_size=params["input_size"], num_class=params["output_size"],
                     hidden_size=params["hidden_size"],
                     num_layers=params["num_layers"])
    else:
        raise Exception("Unknow rnn type:{}".format(params["rnn_type"]))
    return model


def sent2tensor(sent, input_dim, word2idx, wv_matrix, device):
    idx_seq = []
    for w in sent:
        if w in word2idx:
            idx = word2idx[w]
        elif w.lower() in word2idx:
            idx = word2idx[w.lower()]
        else:
            idx = wv_matrix.shape[0] - 1
        idx_seq.append(idx)
    seq = torch.zeros(1, len(idx_seq), input_dim).to(device)
    for i, w_idx in enumerate(idx_seq):
        seq[0][i] = torch.tensor(wv_matrix[w_idx])
    return seq


def load_model(model_type, dataset, device, load_model_path=STANDARD_PATH):
    params = getattr(train_args, "args_{}_{}".format(model_type, dataset))()
    params["rnn_type"] = model_type
    model = init_model(params=params)

    if load_model_path == STANDARD_PATH:
        model_path = get_path(getattr(getattr(TrainedModel, model_type.upper()), dataset.upper()))
    else:
        model_path = get_path(load_model_path)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model


def get_model_file(data_type, model_type):
    if data_type == DateSet.MR:
        if model_type == ModelType.LSTM:
            # model_file = "train_acc-0.7892-test_acc-0.7798.pkl"
            model_file = "train_acc-0.7988-test_acc-0.7746.pkl"
        else:
            assert model_type == ModelType.GRU
            # model_file = "train_acc-0.7813-test_acc-0.7840.pkl"
            model_file = "train_acc-0.8096-test_acc-0.7732.pkl"
    else:
        if model_type == ModelType.LSTM:
            model_file = "train_acc-0.8807-test_acc-0.8435.pkl"
        else:
            assert model_type == ModelType.GRU
            model_file = "train_acc-0.8836-test_acc-0.8424.pkl"
    return model_file
