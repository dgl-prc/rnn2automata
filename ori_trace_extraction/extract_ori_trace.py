from target_models.model_training import sent2tensor
from target_models.model_helper import load_model, get_input_dim
from utils.help_func import *
from utils.constant import *
from data.text_utils import filter_stop_words,STOP_WORDS


def make_ori_trace(model_type, dataset, device, use_clean=False, path_mode=STANDARD_PATH,
                   model_path=STANDARD_PATH):
    """

    model_type:
    dataset:
    device:
    use_clean: bool if true then filter the stop words, or keep the stop words
    :return:
    """
    input_dim = get_input_dim(dataset)

    if dataset.startswith("tomita"):
        gram_id = int(dataset[-1])
        data = load_pickle(get_path(getattr(DataPath, "TOMITA").PROCESSED_DATA).format(gram_id, gram_id))
        wv_matrix = load_pickle(get_path(getattr(DataPath, "TOMITA").WV_MATRIX).format(gram_id, gram_id))
        model = load_model(model_type, "tomita", device=device, load_model_path=model_path)
    else:
        model = load_model(model_type, dataset, device=device, load_model_path=model_path)
        data = load_pickle(get_path(getattr(DataPath, dataset.upper()).PROCESSED_DATA))
        wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX))

    word2idx = data["word_to_idx"]
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
    print("Saved to {}".format(save_path))
