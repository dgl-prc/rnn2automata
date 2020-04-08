import os

PROJECT_ROOT = "/home/dgl/project/learn_automata_rnn"
SENTENCE_ENCODER = "/home/dgl/project/Adversarial-Sampling-for-Repair/text_attack/textbugger/universal-sentence-encoder"
PRISM_SCRIPT = "/home/dgl/project/pfa_extraction/experiments/application_adv/reachability/prism/v-4.5/bin/prism"
WORD2VEC_PATH = "data/wordvec/GoogleNews-vectors-negative300.bin"
PROPERTY_FILE = "experiments/application/find_important_words/properties.pctl"

START_SYMBOL = 'S'
STANDARD_PATH = "STANDARD"


class PartitionType:
    KM = "km"  # kmeans
    KMP = "kmp"  # kmeans based on probas
    HC = "hc"  # hierarchical-clustering


def get_path(r_path):
    return os.path.join(PROJECT_ROOT, r_path)


class DateSet:
    IMDB = "imdb"
    MR = "mr"
    BP = "bp"
    Tomita1 = 1
    Tomita2 = 2
    Tomita3 = 3
    Tomita4 = 4
    Tomita5 = 5
    Tomita6 = 6
    Tomita7 = 7


class ModelType:
    SRNN = 'srnn'
    LSTM = 'lstm'
    GRU = 'gru'


class DataPath:
    class BP:
        PROCESSED_DATA = "data/training_data/bp/bp.pkl"
        WV_MATRIX = "data/training_data/bp/bp_wv_matrix.pkl"

    class TOMITA:
        PROCESSED_DATA = "data/training_data/tomita/tomita_{}/tomita_{}.pkl"
        WV_MATRIX = "data/training_data/tomita/tomita_{}/tomita_{}_wv_matrix.pkl"

    class IMDB:
        RAW_DATA = "data/training_data/imdb/raw"
        PROCESSED_DATA = "data/training_data/imdb/processed_imdb.pkl"
        WV_MATRIX = "data/training_data/imdb/imdb_wv_matrix.pkl"

    class MR:
        RAW_DATA = "data/training_data//mr/raw"
        PROCESSED_DATA = "data/training_data/mr/processed_mr.pkl"
        WV_MATRIX = "data/training_data/mr/mr_wv_matrix.pkl"


class TrainedModel:
    # data/no_stopws/trained_models/{data_type}/{model_type}/{file_name}
    NO_STOPW = get_path("data/no_stopws/trained_models/{}/{}/{}")
    # class LSTM:
    #     IMDB = "data/trained_models/lstm/imdb/"
    #     MR = "data/trained_models/lstm/mr/"
    #     # MR = "data/trained_models/lstm/mr/train_acc-0.8243-test_acc-0.7788.pkl"
    #
    # class GRU:
    #     IMDB = "data/trained_models/gru/imdb/"
    #     MR = "data/trained_models/gru/mr/"


class OriTrace:
    NO_STOPW = get_path("data/no_stopws/ori_trace/{}/{}.pkl")  # data_type/model_type
    # class LSTM:
    #     IMDB = "data/ori_trace/lstm/imdb.pkl"
    #     MR = "data/ori_trace/lstm/mr.pkl"
    #
    # class GRU:
    #     IMDB = "data/ori_trace/gru/imdb.pkl"
    #     MR = "data/ori_trace/gru/mr.pkl"


class AbstractData:
    class Level1:
        # e.g. (data_type, model_type, k, data_source+".txt")
        NO_STOPW = get_path("data/no_stopws/L1_trace/{}/{}/k={}/{}")
        # class KM:
        #     class LSTM:
        #         MR = "data/level1_abs_trace/km/lstm/"
        #
        # class KMP:
        #     class LSTM:
        #         MR = "data/level1_abs_trace/kmp/lstm/"
        #
        # class HC:
        #     class LSTM:
        #         MR = "data/level1_abs_trace/hc/lstm/"

    class Level2:
        NO_STOPW = get_path("data/no_stopws/L2_results/{}/{}/k={}/alpha={}")  # {data_type}/{model_type}/
        # class KM:
        #     class LSTM:
        #         MR = "data/level2_results/km/lstm/"
        #
        # class KMP:
        #     class LSTM:
        #         MR = "data/level2_results/kmp/lstm/"
        #
        # class HC:
        #     class LSTM:
        #         MR = "data/level2_results/hc/lstm/"


class Application:
    class AEs:
        NO_STOPW = get_path("data/no_stopws/adv_text/{}/{}/{}.pkl")  # /{data_type}/{model_type}/{bug_model}.pkl
        #
        # class MR:
        #     LSTM = "data/application/aes/mr/lstm-{}.pkl"  # different mode.
