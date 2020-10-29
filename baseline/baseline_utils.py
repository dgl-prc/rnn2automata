from level2_abstract.read_seq import *
from utils.help_func import load_pickle
from target_models.model_helper import load_model, get_model_file


def prepare_input(data_type, model_type, data_source, k=-1, total_symbols=1000000):
    ############################
    # load model and sentences
    ############################
    model_file = get_model_file(data_type, model_type)
    model_path = get_path(TrainedModel.NO_STOPW.format(data_type, model_type, model_file))
    if data_type.startswith("tomita"):
        gram_id = int(data_type[-1])
        data = load_pickle(get_path(getattr(DataPath, "TOMITA").PROCESSED_DATA).format(gram_id, gram_id))
        model = load_model(model_type, "tomita", device="cpu", load_model_path=model_path)
        wv_matrix = load_pickle(get_path(getattr(DataPath, "TOMITA").WV_MATRIX).format(gram_id, gram_id))
    else:
        model = load_model(model_type, data_type, device="cpu", load_model_path=model_path)
        data = load_pickle(get_path(getattr(DataPath, data_type.upper()).PROCESSED_DATA))
        wv_matrix = load_pickle(get_path(getattr(DataPath, data_type.upper()).WV_MATRIX))

    if k != -1:
        #################################
        # load L1 traces and partitioner
        ##################################
        l1_data_path = AbstractData.Level1.NO_STOPW.format(data_type, model_type, k, data_source + ".txt")
        l1_traces, alphabet = load_trace_data(l1_data_path, total_symbols)
        pt_path = AbstractData.Level1.NO_STOPW.format(data_type, model_type, k, data_source + "_partition.pkl")
        partitioner = load_pickle(pt_path)
        return l1_traces, wv_matrix, data, model, partitioner
    else:
        return wv_matrix, data, model
