import sys
import time

sys.path.append("../../")
from baseline.bl2.icml2018_dfa.Extraction import extract
from baseline.bl2.oracle import Oracle
from baseline.baseline_utils import prepare_input
from target_models.model_helper import get_input_dim
from data.text_utils import is_use_clean, is_artificial, filter_stop_words
from sklearn.utils import shuffle
from baseline.bl2.my_string import MyString
from utils.constant import DateSet, ModelType


def test_accuracy(fa_model, dataset, real_sense):
    '''
    :param model:
    :param dataset:
    :return:
    '''
    correct = 0
    for x, y in zip(dataset["x"], dataset["y"]):
        pdt = fa_model.classify_word(x, real_sense)
        if pdt == y:
            correct += 1
    return correct / len(dataset["y"])


def test_accuracy_rnn(fa_model, dataset):
    '''
    :param model:
    :param dataset:
    :return:
    '''
    correct = 0
    for key in dataset.keys():
        label = dataset[key]
        pdt = fa_model.classify_word(key, -1)
        if pdt == label:
            correct += 1
    return correct / len(dataset)


def test_fidelity(fa_model, rnn, dataset, real_sense):
    count = 0
    for key in dataset:
        fa_pdt = fa_model.classify_word(key, real_sense)
        rnn_pdt = rnn.classify_word(key)
        if fa_pdt == rnn_pdt:
            count += 1
    return count / len(dataset)


def get_start_samples(dataset, rnn):
    all_words = sorted(dataset, key=lambda x: len(x))
    pos = next((w for w in all_words if rnn.classify_word(w) == True), None)
    neg = next((w for w in all_words if rnn.classify_word(w) == False), None)
    starting_examples = [w for w in [pos, neg] if not None == w]
    return starting_examples


def format_data(data_type, data, is_train_data):
    """ convert the list into a string.
    data_type:
    data: list(list).
    Return:
        list(string)
    """
    data_source = "train" if is_train_data else "test"

    if is_artificial(data_type):
        X = ["".join(w) for w in data["{}_x".format(data_source)]]
    else:
        X = []
        for w in data["{}_x".format(data_source)]:
            w = filter_stop_words(w)
            if len(w) != 0:
                X.append(MyString(w))
    return X, data["{}_y".format(data_source)]


def select_data(X, Y, size, random_state=2020):
    X, Y = shuffle(X, Y, random_state=random_state)
    return X[:size], Y[:size]


def extract_alphabet(data):
    alphabet = list(set([w for sent in data for w in sent.data]))
    alphabet.sort(key=lambda x: len(x))
    return alphabet


def test_bl2():
    _data_source = "train"
    device = "cpu"
    for _data_type in [DateSet.BP, DateSet.Tomita1, DateSet.Tomita2, DateSet.Tomita3,
                       DateSet.Tomita4, DateSet.Tomita5,
                       DateSet.Tomita6, DateSet.Tomita7]:
        for _model_type in [ModelType.LSTM, ModelType.GRU]:
            print(_data_type.upper(), _model_type.upper())
            _use_clean = is_use_clean(_data_type)
            input_dim = get_input_dim(_data_type)
            wv_matrix, data, model, = prepare_input(_data_type, _model_type, _data_source)
            train_x, train_y = format_data(_data_type, data, is_train_data=True)
            alphabet = data["vocab"]
            if is_artificial(_data_type):
                alphabet = alphabet[1:]
                real_sent = False
                test_x, test_y = format_data(_data_type, data, is_train_data=False)
            else:
                data_size = int(sys.argv[3])
                train_x, train_y = select_data(train_x, train_y, size=data_size)
                alphabet = extract_alphabet(train_x)
                real_sent = True
                test_x, test_y = train_x, train_y
            rnn = Oracle(_data_type, _model_type, model, alphabet, _model_type, input_dim, data["word_to_idx"],
                         wv_matrix,
                         device)
            starting_examples = get_start_samples(train_x, rnn)
            start_clock = time.clock()
            dfa = extract(rnn, time_limit=400, initial_split_depth=10, starting_examples=starting_examples,
                          real_sense=real_sent)
            total_time = time.clock() - start_clock
            ###########
            # test data
            ###########
            acc = test_accuracy(dfa, {"x": test_x, "y": test_y}, real_sense=False)
            fdlt = test_fidelity(dfa, rnn, test_x, real_sense=False)
            print("DFA,Accuracy:{:.4f},Fidelity:{:.4f}, DFA size:{}, alphabet size:{},time:{}".format(acc, fdlt,
                                                                                                      len(dfa.Q),
                                                                                                      len(alphabet),
                                                                                                      total_time))

if __name__ == "__main__":
    test_bl2()
