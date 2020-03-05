from ori_trace_extraction.extract_ori_trace import *


def valid_pre_y():
    data = load_pickle(get_path(DataPath.MR.PROCESSED_DATA))
    ori_traces = load_pickle(get_path(OriTrace.LSTM.MR))
    predicts = ori_traces["test_pre_y"]
    labels = data["test_y"]
    cnt = 0
    for pre_y, y in zip(predicts, labels):
        if y == pre_y:
            cnt += 1
    print(cnt / len(labels))

if __name__ == '__main__':
    make_ori_trace(ModelType.LSTM, DateSet.MR, "cuda:0")
    # valid_pre_y()
