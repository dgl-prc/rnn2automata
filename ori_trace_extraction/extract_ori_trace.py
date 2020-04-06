from target_models.model_training import sent2tensor
from target_models.model_helper import load_model
from utils.help_func import *
from utils.constant import *
from data.text_utils import filter_stop_words


def make_ori_trace(model_type, dataset, device, use_clean=False, path_mode=STANDARD_PATH,
                   load_model_path=STANDARD_PATH):
    """

    model_type:
    dataset:
    device:
    use_clean: bool if true then filter the stop words, or keep the stop words
    :return:
    """
    data = load_pickle(get_path(getattr(DataPath, dataset.upper()).PROCESSED_DATA))
    word2idx = data["word_to_idx"]
    wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX))
    input_dim = 300

    model = load_model(model_type, dataset, device, load_model_path)

    ori_traces = {}
    ori_traces["train_x"] = []
    ori_traces["test_x"] = []
    ori_traces["train_pre_y"] = []
    ori_traces["test_pre_y"] = []
    print("do extracting...")
    for x in data["train_x"]:
        if use_clean:
            x = filter_stop_words(x)
        tensor_sequence = sent2tensor(x, input_dim, word2idx, wv_matrix, device)
        hn_trace, label_trace = model.get_predict_trace(tensor_sequence)
        ori_traces["train_x"].append(hn_trace)
        ori_traces["train_pre_y"].append(label_trace[-1])

    for x in data["test_x"]:
        if use_clean:
            x = filter_stop_words(x)
        tensor_sequence = sent2tensor(x, input_dim, word2idx, wv_matrix, device)
        hn_trace, label_trace = model.get_predict_trace(tensor_sequence)
        ori_traces["test_x"].append(hn_trace)
        ori_traces["test_pre_y"].append(label_trace[-1])

    if path_mode == STANDARD_PATH:
        save_path = get_path(getattr(getattr(OriTrace, model_type.upper()), dataset.upper()))
    else:
        save_path = path_mode

    save_pickle(save_path, ori_traces)
    print("Saved!")
