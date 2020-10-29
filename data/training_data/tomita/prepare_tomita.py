import random
import numpy as np
from data.training_data.tomita import tomita_fa
from utils.constant import *
from utils.help_func import save_pickle


class Generator(object):
    '''
    For each length, we generate the identical number of positive strings and negative strings.
    '''

    def __init__(self, grammarType, len_pool, max_iteries, max_size_per_len, ub_neg_num=20):
        '''
        :param grammarType: tomita grammar
        :param len_pool:list. the list of the string's length.
        :param max_iteries: the max number of iterations trying to randomly generate a desired sample.
        :param max_size_per_len: the max number of samples for each length.
        :param ub_neg_num: the number of negative samples per length under tomita1 and tomita2
        '''
        self.grammarType = grammarType
        self.len_pool = len_pool
        self.max_iteries = max_iteries
        self.max_size_per_len = max_size_per_len
        self.ub_neg_num = ub_neg_num
        self.alphabet = ["0", "1"]

        if self.grammarType == DateSet.Tomita1:
            self.dfa = tomita_fa.Tomita_1()

        if self.grammarType == DateSet.Tomita2:
            self.dfa = tomita_fa.Tomita_2()

        if self.grammarType == DateSet.Tomita3:
            self.dfa = tomita_fa.Tomita_3()

        if self.grammarType == DateSet.Tomita4:
            self.dfa = tomita_fa.Tomita_4()

        if self.grammarType == DateSet.Tomita5:
            self.dfa = tomita_fa.Tomita_5()

        if self.grammarType == DateSet.Tomita6:
            self.dfa = tomita_fa.Tomita_6()

        if self.grammarType == DateSet.Tomita7:
            self.dfa = tomita_fa.Tomita_7()

    def generate_balance(self):
        data = {}
        for l in self.len_pool:
            for i in range(self.max_size_per_len):
                if l == 0:
                    data[""] = True
                    break
                if self.dfa.get_re() == "1*" or self.dfa.get_re() == "(10)*":
                    if i >= self.ub_neg_num:
                        break
                pos = self.dfa.generatePos(l, data.keys(), self.max_iteries)
                neg = self.dfa.generateNeg(l, data.keys(), self.max_iteries)
                if pos not in data.keys() and len(pos) > 0:
                    data[pos] = True
                if neg not in data.keys() and len(neg) > 0:
                    data[neg] = False
        return data

    def generate_random(self, data_size, rnd_state=2020):
        data = {}
        random.seed(rnd_state)
        for i in range(data_size):
            while True:
                l = random.choice(self.len_pool)
                seq, label = self.dfa.random_word(l, start=self.dfa.get_start(), alphabet=self.dfa.get_alphabet())
                if seq not in data.keys():
                    if label is True:
                        data[seq] = True
                    else:
                        data[seq] = False
                    break
        return data

    # def save_data(self, save_path, data):
    #     '''
    #     save data into a pickle file: {"data":{"str":lable},"num_pos":int,"num_neg":int}
    #     :param save_path:
    #     :param pos_list:
    #     :param neg_list:
    #     :return:
    #     '''
    #     with open(save_path, "wb") as f:
    #         num_pos = len([seq for seq in data.keys() if data[seq] == True])
    #         num_neg = len([seq for seq in data.keys() if data[seq] == False])
    #         assert num_neg + num_pos == len(data)
    #         print("{}({})".format(len(data), num_pos))
    #         pickle.dump({"data": data,
    #                      "num_pos": num_pos,
    #                      "num_neg": num_neg}, f)


def make_wv_matrix_tomita(data):
    index_map = {}
    index_map[''] = 0
    index_map["0"] = 1
    index_map["1"] = 2
    wv_matrix = []
    input_dim = 3  # alphabet_size
    for i in range(len(data["vocab"])):
        word = data["idx_to_word"][i]
        vector = [0.] * input_dim
        vector[index_map[word]] = 1.
        wv_matrix.append(vector)
    wv_matrix = np.array(wv_matrix)
    return wv_matrix


def divide_tomita(tomita_x):
    max_iteries = 100
    max_size_per_len = 150  # 150*2 =300 negative(150) + positive(150)
    ub_neg_num = 50  # number of unbalance negative

    ###############
    # train data
    ##############
    train_len_pool = [i for i in range(14)]  # 0-13
    if tomita_x == DateSet.Tomita6:
        # 0-13,15-20
        train_len_pool.extend([15, 16, 17, 18, 19, 20])
    else:
        # 0-13,16,19,22
        train_len_pool.extend([16, 19, 22])
    g_train = Generator(tomita_x, train_len_pool, max_iteries, max_size_per_len, ub_neg_num)
    train_data = g_train.generate_balance()

    ###############
    # test data
    ##############
    test_data_size = len(train_data) // 4
    test_len_pool = range(1, 29, 3)
    g_test = Generator(tomita_x, test_len_pool, max_iteries, max_size_per_len, ub_neg_num)
    ######################################################################
    # The positive to negative sample ratios in the dev sets were not
    # controlled according to Gail Weiss,ICML18
    ######################################################################
    test_data = g_test.generate_random(test_data_size)

    train_x = []
    train_y = []
    for word in train_data:
        train_x.append([w for w in word] if word != "" else [''])
        train_y.append(int(train_data[word]))

    test_x = []
    test_y = []
    for word in test_data:
        test_x.append([w for w in word] if word != "" else [''])
        test_y.append(int(test_data[word]))

    #####################################################
    # compensate tomita-2 in case of no positive samples
    #####################################################
    if tomita_x == DateSet.Tomita2  and sum(test_y) == 0:
        # 1, 4, 7, 10, 13, 16, 19, 22, 25, 28
        pos1 = ['']
        pos2 = [w for w in '1010']  # 4
        pos3 = [w for w in '1010101010']  # 10
        pos4 = [w for w in '1010101010101010']  # 16
        pos5 = [w for w in '1010101010101010101010']  # 22
        pos6 = [w for w in '1010101010101010101010101010']  # 28
        test_x.extend([pos1, pos2, pos3, pos4, pos5, pos6])
        test_y.extend([1]*6)


    data = {}
    data["train_x"], data["test_x"] = train_x, test_x
    data["train_y"], data["test_y"] = train_y, test_y
    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["test_x"] for w in sent])))
    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}
    wv_matrix = make_wv_matrix_tomita(data)

    save_path = get_path(DataPath.TOMITA.PROCESSED_DATA).format(tomita_x[-1], tomita_x[-1])
    save_wv_matrix_path = get_path(DataPath.TOMITA.WV_MATRIX).format(tomita_x[-1], tomita_x[-1])
    print("train: {}({}), test:{}({})".format(len(data["train_y"]), sum(data["train_y"]), len(data["test_y"]),
                                              sum(data["test_y"])))
    save_pickle(save_path, data)
    save_pickle(save_wv_matrix_path, wv_matrix)
    print("data saved in {}".format(save_path))
    print("wv_matrx saved in {}".format(save_wv_matrix_path))
