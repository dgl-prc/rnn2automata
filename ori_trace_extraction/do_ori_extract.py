import sys

sys.path.append("../")
from ori_trace_extraction.extract_ori_trace import *
from target_models.model_helper import get_model_file
from data.text_utils import is_use_clean


def valid_pre_y(ori_traces_path):
    data = load_pickle(get_path(DataPath.MR.PROCESSED_DATA))
    ori_traces = load_pickle(ori_traces_path)
    predicts = ori_traces["test_pre_y"]
    labels = data["test_y"]
    cnt = 0
    for pre_y, y in zip(predicts, labels):
        if y == pre_y:
            cnt += 1
    print(cnt / len(labels))


if __name__ == '__main__':
    data_type = sys.argv[1]
    model_type = sys.argv[2]
    device_id = int(sys.argv[3])

    use_clean = is_use_clean(data_type)
    _device = "cuda:{}".format(device_id) if device_id >= 0 else "cpu"
    model_file = get_model_file(data_type, model_type)
    save_path = OriTrace.NO_STOPW.format(data_type, model_type)
    model_path = get_path(TrainedModel.NO_STOPW.format(data_type, model_type, model_file))
    make_ori_trace(model_type, data_type, _device, use_clean=use_clean, path_mode=save_path, model_path=model_path)
    # valid_pre_y(save_path)
    # get_path(OriTrace.LSTM.MR)
