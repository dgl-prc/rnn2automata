import os

PROJECT_ROOT = "/home/dgl/project/learn_automata_rnn"
SENTENCE_ENCODER = "/home/dgl/project/Adversarial-Sampling-for-Repair/text_attack/textbugger/universal-sentence-encoder"
WORD2VEC_PATH = "data/wordvec/GoogleNews-vectors-negative300.bin"
PRISM_DATA_PATH = ""
PROPERTY_FILE = ""
PRISM_SCRIPT = ""
START_SYMBOL = 'S'


class PartitionType:
    KM = "KM"  # kmeans
    HC = "hc" # hierarchical-clustering


def get_path(r_path):
    return os.path.join(PROJECT_ROOT, r_path)


class DateSet:
    IMDB = "imdb"
    MR = "mr"


class ModelType:
    SRNN = 'srnn'
    LSTM = 'lstm'
    GRU = 'gru'


class DataPath:
    class IMDB:
        RAW_DATA = "data/training_data/imdb/raw"
        PROCESSED_DATA = "data/training_data/imdb/processed_mr.pkl"
        WV_MATRIX = "data/training_data/imdb/mr_wv_matrix.pkl"

    class MR:
        RAW_DATA = "data/training_data//mr/raw"
        PROCESSED_DATA = "data/training_data/mr/processed_mr.pkl"
        WV_MATRIX = "data/training_data/mr/mr_wv_matrix.pkl"


class TrainedModel:
    class LSTM:
        IMDB = "data/trained_models/lstm/imdb/"
        MR = "data/trained_models/lstm/mr/train_acc-0.8243-test_acc-0.7788.pkl"

    class GRU:
        IMDB = ""
        MR = ""


class OriTrace:
    class LSTM:
        IMDB = "data/ori_trace/lstm/imdb.pkl"
        MR = "data/ori_trace/lstm/mr.pkl"

    class GRU:
        IMDB = "data/ori_trace/gru/imdb.pkl"
        MR = "data/ori_trace/gru/mr.pkl"


class AbstractData:
    class Level1:
        class KM:
            class LSTM:
                MR = "data/level1_abs_trace/km/lstm/"

        class HC:
            class LSTM:
                MR = "data/level1_abs_trace/hc/lstm/"

    class Level2:
        class KM:
            class LSTM:
                MR = "data/level2_results/km/lstm/"

        class HC:
            class LSTM:
                MR = "data/level2_results/hc/lstm/"


class Application:
    class AEs:
        class MR:
            LSTM = "data/application/aes/mr/lstm.pkl"
