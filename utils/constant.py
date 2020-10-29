import os

PROJECT_ROOT = "/home/dgl/project/learn_automata_rnn"
SENTENCE_ENCODER = "/home/dgl/project/Adversarial-Sampling-for-Repair/text_attack/textbugger/universal-sentence-encoder"
PRISM_SCRIPT = "/home/dgl/project/pfa_extraction/experiments/application_adv/reachability/prism/v-4.5/bin/prism"
WORD2VEC_PATH = "data/wordvec/GoogleNews-vectors-negative300.bin"
PROPERTY_FILE = "experiments/effectiveness/properties.pctl"

START_SYMBOL = 'S'
STANDARD_PATH = "STANDARD"

def get_total_symbols(data_set):
    """
    return the total symbols used for learning PFA
    :param data_set:
    :return:
    """
    if data_set == DateSet.MR:
        return 107355
    if data_set == DateSet.IMDB:
        return 141953
    if data_set == DateSet.BP:
        return 54044
    if data_set == DateSet.Tomita1:
        return 8290
    if data_set == DateSet.Tomita2:
        return 8221
    if data_set == DateSet.Tomita3:
        return 42879
    if data_set == DateSet.Tomita4:
        return 43009
    if data_set == DateSet.Tomita5:
        return 34009
    if data_set == DateSet.Tomita6:
        return 59009
    if data_set == DateSet.Tomita7:
        return 42659
    return -1

class PartitionType:
    KM = "km"  # kmeans
    KMP = "kmp"  # kmeans based on probas
    HC = "hc"  # hierarchical-clustering


def get_path(r_path):
    return os.path.join(PROJECT_ROOT, r_path)


class DateSet:
    IMDB = "imdb"
    MR = "mr"
    # arificail data
    BP = "bp"
    Tomita1 = "tomita1"
    Tomita2 = "tomita2"
    Tomita3 = "tomita3"
    Tomita4 = "tomita4"
    Tomita5 = "tomita5"
    Tomita6 = "tomita6"
    Tomita7 = "tomita7"

class ModelType:
    SRNN = 'srnn'
    LSTM = 'lstm'
    GRU = 'gru'

class DataPath:
    class BP:
        PROCESSED_DATA = "data/training_data/bp/bp.pkl"
        WV_MATRIX = "data/training_data/bp/bp_wv_matrix.pkl"

    class TOMITA:
        PROCESSED_DATA = "data/training_data/tomita/tomita{}/tomita{}.pkl"
        WV_MATRIX = "data/training_data/tomita/tomita{}/tomita{}_wv_matrix.pkl"

    class IMDB:
        RAW_DATA = "data/training_data/imdb/raw"
        PROCESSED_DATA = "data/training_data/imdb/processed_imdb.pkl"
        WV_MATRIX = "data/training_data/imdb/imdb_wv_matrix.pkl"

    class MR:
        RAW_DATA = "data/training_data//mr/raw"
        PROCESSED_DATA = "data/training_data/mr/processed_mr.pkl"
        WV_MATRIX = "data/training_data/mr/mr_wv_matrix.pkl"


class TrainedModel:
    NO_STOPW = get_path("data/no_stopws/trained_models/{}/{}/{}")

class OriTrace:
    NO_STOPW = get_path("data/no_stopws/ori_trace/{}/{}.pkl")  # data_type/model_type

class AbstractData:
    class Level1:
        NO_STOPW = get_path("data/no_stopws/L1_trace/{}/{}/k={}/{}")

    class Level2:
        NO_STOPW = get_path("data/no_stopws/L2_results/{}/{}/k={}/alpha={}")  # {data_type}/{model_type}/

class Application:
    class AEs:
        NO_STOPW = get_path("data/no_stopws/adv_text/{}/{}/{}.pkl")  # /{data_type}/{model_type}/{bug_model}.pkl

