"""
Since some models on artificial dataset can not make right prediction on any of positive
samples due to the imbalanced training set, we retrain those model to address that by data augmentation.
"""
import sys

sys.path.append("../")
from os.path import abspath
from target_models import train_args
from ori_trace_extraction.extract_ori_trace import *
from target_models.model_helper import get_model_file
from data.text_utils import is_use_clean
from target_models.model_training import train
from utils.time_util import *

def data_augmentation(data, scale):
    positives_x = []
    positives_y = []
    for x, y in zip(data["train_x"], data["train_y"]):
        if y == 1:
            positives_x.append(x)
            positives_y.append(y)
    size_neg = len(data["train_y"]) - sum(data["train_y"])
    repeat_times = int(size_neg * scale / len(positives_y)) - 1
    for i in range(repeat_times):
        data["train_x"].extend(positives_x)
        data["train_y"].extend(positives_y)
    return data


if __name__ == '__main__':
    dataset = sys.argv[1]
    model_type = sys.argv[2]
    gpu = int(sys.argv[3])
    aug_scale = 0.5
    _device = "cuda:{}".format(gpu) if gpu >= 0 else "cpu"
    use_clean = is_use_clean(dataset)
    assert dataset.startswith("tomita")
    gram_id = int(dataset[-1])
    params = getattr(train_args, "args_{}_{}".format(model_type, "tomita"))()
    data = load_pickle(get_path(getattr(DataPath, "TOMITA").PROCESSED_DATA).format(gram_id, gram_id))
    wv_matrix = load_pickle(get_path(getattr(DataPath, "TOMITA").WV_MATRIX).format(gram_id, gram_id))
    _data = data_augmentation(data, aug_scale)
    model_file = get_model_file(dataset, model_type)
    model_path = get_path(TrainedModel.NO_STOPW.format(dataset, model_type, model_file))
    _model = load_model(model_type, "tomita", device=_device, load_model_path=model_path)

    #######################
    # load training params
    #######################
    params = getattr(train_args, "args_{}_{}".format(model_type, "tomita"))()
    train_args.add_data_info(data, params)
    params["WV_MATRIX"] = wv_matrix
    params["device"] = _device
    params["rnn_type"] = model_type
    params["use_clean"] = use_clean
    params["EPOCH"] = 200
    ################
    # retraining
    ################
    model, train_acc, test_acc = train(_model, _data, params)
    # save model
    save_folder = "data/no_stopws/trained_models/{}/{}/".format(dataset, model_type)
    save_path = os.path.join(PROJECT_ROOT, save_folder, folder_timestamp())
    save_model(model, train_acc, test_acc, abspath(save_path))
    save_readme(save_path, ["{}:{}\n".format(key, params[key]) for key in params.keys() if key != "WV_MATRIX"])
    print("model saved to {}".format(save_path))

