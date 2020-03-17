import sys
import copy
import gensim

sys.path.append('../../../')
from target_models.classifier_adapter import Classifier
from target_models.model_helper import load_model, sent2tensor
from level1_abstract.clustering_based import *
from experiments.exp_utils import *
from experiments.application.find_important_words.get_reachability_matrix import reachability_matrix
from experiments.application.find_important_words.sort_words import sort_words_by_rm, get_word_score
from experiments.application.adv_detect.textbugger.textbugger_attack import TextBugger



predict_map = {0: "N", 1: "P"}
class Sent2Trace():
    def __init__(self, model, input_dim, word2idx, wv_matrix, device):
        self.model = model
        self.input_dim = input_dim
        self.word2idx = word2idx
        self.wv_matrix = wv_matrix
        self.device = device

    def get_rnn_trace(self, new_sent):
        sent_tensor = sent2tensor(new_sent, self.input_dim, self.word2idx, self.wv_matrix, self.device)
        hn_trace, label_trace = self.model.get_predict_trace(sent_tensor)
        return hn_trace, label_trace[-1]

    def get_l1_trace(self, new_sent, partitioner):
        hn_trace, predict = self.get_rnn_trace(new_sent)
        l1_trace = level1_abstract(rnn_traces=[hn_trace], y_pre=[predict], partitioner=partitioner,
                                   partitioner_exists=True)[0]
        return l1_trace, predict


class MakeSynonyms(object):

    def __init__(self, word2vec, topn, sent_factory, partitioner, trans_func, reach_matrix):
        self.word2vec = word2vec
        self.topn = topn
        self.sent_factory = sent_factory
        self.partitioner = partitioner
        self.trans_func = trans_func
        self.reach_matrix = reach_matrix

    def replace_word(self, sent, w_idx, old_score, old_y):

        need_more_replace = True
        target_word = sent[w_idx]
        if target_word not in self.word2vec.vocab:
            return sent, need_more_replace
        words_list = self.word2vec.most_similar(positive=[target_word], topn=self.topn)
        bugs = [item[0] for item in words_list]
        max_score = 0
        best_new_sent = sent
        for bug in bugs:
            new_sent = copy.deepcopy(sent)
            new_sent[w_idx] = bug
            l1_trace, predict = self.sent_factory.get_l1_trace(new_sent, self.partitioner)
            if predict != old_y:
                # attack succeeds
                need_more_replace = False
                best_new_sent = new_sent
                break
            reach_probs_list = get_reach_probs_list(l1_trace, self.trans_func, self.reach_matrix)
            if len(l1_trace) - len(reach_probs_list) == 2:
                assert len(sent) == len(reach_probs_list)
                new_reach_probs = reach_probs_list[w_idx]
                new_score = get_word_score(desired_y=old_y, reach_probs=new_reach_probs)
                significance = old_score - new_score
                if significance > max_score:
                    max_score = significance
                    best_new_sent = new_sent
            else:
                # exists unspecified transitions
                pass
        return best_new_sent, need_more_replace


def get_reach_probs_list(l1_trace, trans_func, reach_matrix):
    c_id = 1  # current state id
    reach_probs_list = []
    for sigma in l1_trace[1:-1]:
        sigma = str(sigma)
        if sigma not in trans_func[c_id]:
            break
        else:
            next_id = trans_func[c_id][sigma]
            reach_probs = reach_matrix[next_id - 1]
            reach_probs_list.append(reach_probs)
    return reach_probs_list


def attack(**kwargs):
    input_dim = 300
    device = "cpu"
    model_type = kwargs["m_tp"]
    data_type = kwargs["d_tp"]
    data_source = kwargs["d_s"]
    pt_type = kwargs["pt_tp"]
    total_symbols = kwargs["t_syms"]
    k = kwargs["k"]
    num_prop = kwargs["num_prop"]

    pm_file = getattr(getattr(getattr(AbstractData.Level2, pt_type.upper()), model_type.upper()), data_type.upper())
    pm_file = os.path.join(get_path(pm_file), data_source,
                           "{}_{}_k{}_{}.pm".format(model_type, data_type, k, total_symbols))

    trans_func, trans_wfunc = load_dfa(model_type, data_type, k, total_symbols, data_source, pt_type)
    reach_matrix = reachability_matrix(pm_file, num_prop)

    #####################
    # load partitioner
    ####################
    partitioner = load_partitioner(model_type, data_type, pt_type, k, data_source)

    #############
    # load data
    #############
    raw_data = load_pickle(get_path(getattr(DataPath, data_type.upper()).PROCESSED_DATA))
    wv_matrix = load_pickle(get_path(getattr(DataPath, data_type.upper()).WV_MATRIX))
    word2idx = raw_data["word_to_idx"]

    ############
    # load model
    ############
    model = load_model(model_type, data_type, device)
    classifier = Classifier(model, model_type, input_dim, raw_data["word_to_idx"], wv_matrix, device)

    # select benign
    benign_tuple, benign_idx = select_benign_data(classifier, raw_data)
    benign_labels = [ele[1] for ele in benign_tuple]
    benign_data = [ele[0] for ele in benign_tuple]
    ###################
    # extract ori trace
    ###################
    sent_factory = Sent2Trace(model, input_dim, word2idx, wv_matrix, device)
    benign_traces = []
    for benign_x in benign_data:
        benign_hn_trace, predict = sent_factory.get_rnn_trace(benign_x)
        benign_traces.append(benign_hn_trace)

    benign_l1_trace = level1_abstract(rnn_traces=benign_traces, y_pre=benign_labels, partitioner=partitioner,
                                      partitioner_exists=True)

    #############
    # do attack
    #############
    print("attacking....")
    topn = 5
    word2vec_model_path = get_path(WORD2VEC_PATH)
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

    textbugger = TextBugger(classifier, word2vec)
    makeSynonyms = MakeSynonyms(word2vec, topn, sent_factory, partitioner, trans_func, reach_matrix)

    attack_cnt = 0
    attack_cnt_tb = 0 # textbugger
    p = 0
    for sent, y_true, l1_trace in zip(benign_data[:1000], benign_labels[:1000], benign_l1_trace[:1000]):
        p += 1
        ###########################
        # reachbility based attack
        ###########################
        sorted_words, sorted_idx, score_list = sort_words_by_rm(sent, l1_trace, reach_matrix, trans_func)
        new_sent = sent
        y_label = predict_map[y_true]
        for w_idx, old_score in zip(sorted_idx, score_list):
            new_sent, is_continue = makeSynonyms.replace_word(new_sent, w_idx, old_score, y_label)
            if not is_continue:
                sim = textbugger.similarity(sent, new_sent)
                if sim > textbugger.sim_epsilon:
                    attack_cnt += 1
                    break
        ###############
        # textbugger
        ###############
        _, rst = textbugger.attack(sent)
        if rst != -1:
            attack_cnt_tb += 1

        sys.stdout.write("\r progress: {:.2f}%".format(100 * p / 1000))
        sys.stdout.flush()
    print("\n")
    print("Total attack success :{:.2f}% (Reachbility Based), {:.2f}% (TextBugger)".format(100 * attack_cnt / 1000, 100 * attack_cnt_tb / 1000))


if __name__ == '__main__':
    _num_prop = 2
    _k = 10
    _model_type = ModelType.LSTM
    _data_type = DateSet.MR
    _data_source = "train"
    _total_symbols = 196425
    _pt_type = PartitionType.KM
    print("***********k={}**************".format(_k))
    attack(m_tp=_model_type, d_tp=_data_type, d_s=_data_source, pt_tp=_pt_type, t_syms=_total_symbols, k=_k,
          num_prop=_num_prop)