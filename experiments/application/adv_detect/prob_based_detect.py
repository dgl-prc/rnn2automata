import sys

sys.path.append("../../")
from utils.constant import *
from pfa_build.pfa import PFA
from utils.time_util import *
from utils.save_utils import *
from utils.adhoc_io import *
from experiments.application_adv.adversaries_generation import MRClassifier, select_benign_data
from rnn_models import train_args
from sklearn.metrics import roc_curve, auc, accuracy_score
import random
from pfa_extractor.trace_processor import *
from pfa_build.abs_trace_extractor import AbstractTraceExtractor
from data.process_utils import RealDataProcessor
import math
import warnings


def make_y_scores(pos, neg):
    '''
    Note the pos item should have a bigger value, while neg should have a smaller.
    '''
    assert isinstance(pos, list)
    assert isinstance(neg, list)
    scores = pos + neg
    y_ture = [1] * len(pos) + [0] * len(neg)
    return y_ture, scores


def get_auc(pos_score, neg_score):
    y_ture, y_scores = make_y_scores(pos=pos_score, neg=neg_score)
    fpr, tpr, thresholds = roc_curve(y_ture, y_scores)
    auc_score = auc(fpr, tpr)
    return auc_score


def get_pprob(is_use_word_trace, trace_processor, pfa, sent, trans):
    ''' get path probability

    Parameters
    ----------------
    is_use_word_trace: bool. True: get path probability with sentence directly. False get the path probability with action trace
    trace_processor:
    pfa:
    sent: list. in which each element is a word of the sentence.
    trans: the state transition matrix.
    Returns:
    '''
    if is_use_word_trace:
        predict, path_prob, is_terminate, state_trans_path = pfa.predict_word_trace(sent, trans)
    else:
        action_trace = trace_processor.get_action_trace(sent)
        predict, path_prob, is_terminate, state_trans_path = pfa.predict_with_abs_trace(action_trace=action_trace)
    return path_prob, is_terminate


def prob_normalize(prob, n):
    # with warnings.catch_warnings():
    #     try:
    #         rst = np.log(prob ** (1. / n))
    #     except RuntimeWarning:
    #         print(prob,n)

    return -10 ** 10 if prob == 0.0 else np.log(prob ** (1. / n))


def detect_adv(is_use_word_trace, trace_processor, classifier, data, pfa_id, model_type, dataType, max_length):
    benign_data, benign_idx = select_benign_data(classifier, data)
    ########################
    # prepare pfa
    #######################
    PFA_SAVE_DIR = os.path.join(get_path(getattr(getattr(PDFADataPath, dataType.upper()), model_type.upper())), pfa_id)
    output_path = os.path.join(PFA_SAVE_DIR, "dtmc")
    pm_file_path = os.path.join(output_path, model_type + str(max_length) + ".pm")
    pfa_label_path = os.path.join(output_path, model_type + str(max_length) + "_label.txt")
    pfa = PFA(pm_file_path, pfa_label_path)

    used_traces_path = os.path.join(PFA_SAVE_DIR, "dtmc/used_trace_list.txt")
    trace_path = os.path.join(PFA_SAVE_DIR, "input_trace")
    word_traces_path = os.path.join(PFA_SAVE_DIR, "words_trace")
    trans = pfa.make_words_trans_matrix(used_traces_path=used_traces_path, trace_path=trace_path,
                                        word_traces_path=word_traces_path)
    # benign
    benign_prob = []
    benign_termimate_cnt = 0
    for sent, label in benign_data:
        path_prob, is_terminate = get_pprob(is_use_word_trace, trace_processor, pfa, sent, trans)
        benign_prob.append(prob_normalize(path_prob, len(sent)))
        # benign_prob.append(path_prob)
        if is_terminate:
            benign_termimate_cnt += 1
    # wl
    wl_prob = []
    for idx in range(len(data["test_x"])):
        if idx not in benign_idx:
            sent, label = data["test_x"][idx], data["test_y"][idx]
            path_prob, is_terminate = get_pprob(is_use_word_trace, trace_processor, pfa, sent, trans)
            wl_prob.append(prob_normalize(path_prob, len(sent)))
            # wl_prob.append(path_prob)
    random.seed(20191229)
    auc_score = get_auc(pos_score=random.sample(benign_prob, len(wl_prob)), neg_score=wl_prob)
    print("auc of benign-wl", auc_score)

    # adversarial data
    adv_prob = []
    adv_termimate_cnt = 0
    adv_data = load_pickle(get_path(getattr(getattr(AdvDataPath, dataType.upper()), model_type.upper()).ADV_TEXT))
    for idx, sent, label in adv_data:
        path_prob, is_terminate = get_pprob(is_use_word_trace, trace_processor, pfa, sent, trans)
        adv_prob.append(prob_normalize(path_prob, len(sent)))
        if is_terminate:
            adv_termimate_cnt += 1
    random.seed(20191229)
    auc_score = get_auc(pos_score=random.sample(benign_prob, len(adv_prob)), neg_score=adv_prob)
    print("auc of benign-adv", auc_score)


def do_detect(pfa_id, data_type, model_type):
    # pfa_id = ""
    # model_type = "lstm"
    # data_type = "imdb"

    random_seed = 20191223
    # data_size = 2134
    # data_size = 1000
    input_dim = 300
    device = "cpu"
    max_length = 60000
    is_use_word_trace = False

    params = getattr(train_args, "args_{}_{}".format(model_type.lower(), data_type))()
    params["model_path"] = get_path(getattr(getattr(ModelPath, data_type.upper()), model_type.upper()))
    params["rnn_type"] = model_type
    model = load_model(params=params)
    model = model.to(device)
    model.eval()
    extractor = AbstractTraceExtractor()
    ##################
    # loading data
    #################
    data = load_pickle(get_path(getattr(DataPath, data_type.upper()).PROCESSED_DATA))
    wv_matrix = load_pickle(get_path(getattr(DataPath, data_type.upper()).WV_MATRIX))
    data_processor = RealDataProcessor(device, data["word_to_idx"], params["input_size"], wv_matrix)
    trace_processor = TraceProcessor(extractor, model, None, data_processor, input_dim)
    PFA_SAVE_DIR = os.path.join(get_path(getattr(getattr(PDFADataPath, data_type.upper()), model_type.upper())), pfa_id)
    trace_processor.cluster_model = load_pickle(os.path.join(PFA_SAVE_DIR, "cluster_model.pkl"))
    classifier = MRClassifier(model, device, data["vocab"], data["idx_to_word"], data["word_to_idx"],
                              params["input_size"], wv_matrix)
    detect_adv(is_use_word_trace, trace_processor, classifier, data, pfa_id, model_type, data_type, max_length)


if __name__ == '__main__':

    # data_type = sys.argv[1]
    # model_type = sys.argv[2]
    data_type = "imdb"
    model_type = "gru"
    if data_type == "imdb":
        if model_type == "gru":
            pfa_ids = ["20191228085426", "20191227180357", "20191227181009", "20191227181655", "20191227182358",
                       "20191227184415"]
        else:
            pfa_ids = ["20191227200245", "20191227200748", "20191227201313", "20191227201907", "20191227203258",
                       "20191227205034", "20191227211006"]
    elif data_type == "mr":
        if model_type == "gru":
            pfa_ids = ["20191227163431", "20191227163602", "20191227163738", "20191227163918", "20191227164105",
                       "20191227164310", "20191227164520"]
        else:
            pfa_ids = ["20191227152204", "20191227152531", "20191227152910", "20191227153106", "20191227153426",
                       "20191227154808", "20191227155127"]
    else:
        # spam
        if model_type == "gru":
            pfa_ids = ["20191228224824", "20191228225457", "20191228230225", "20191228231030", "20191228231811",
                       "20191228232556", "20191228233328"]
        else:
            pfa_ids = ["20191228234137", "20191228234712", "20191228235314", "20191228235834", "20191229000441",
                       "20191229001132", "20191229001820"]

    for pfa_id in pfa_ids[5:]:
        print("=============={}==============".format(pfa_id))
        do_detect(pfa_id, data_type, model_type)
