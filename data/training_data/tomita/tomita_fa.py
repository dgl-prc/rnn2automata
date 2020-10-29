# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:43:00 2018
@author: DrLC
"""

import copy
import random
import abc
import math


class DFAState(object):

    def __init__(self, idx=-1, acc=True):
        self.__idx = idx
        self.__acc = acc  # accept
        self.__nxt = {}

    def get_idx(self):
        return self.__idx

    def get_acc(self):
        return self.__acc

    def set_acc(self, acc):
        self.__acc = acc

    def get_nxt(self, key):  # get the next state according to the action(key)
        assert key in self.__nxt.keys(), ("'%s' not in alphabet" % key)
        return self.__nxt[key]

    def set_nxt(self, key, s):
        '''
        :param key: action
        :param s: the next sate leading to
        :return:
        '''
        self.__nxt[key] = s


class Tomita(object):

    def classify(self, seq, start):
        s = start
        for ch in seq:
            s = s.get_nxt(ch)
        rst = s.get_acc()
        return rst

    def random_word(self, l, start, alphabet):
        seq = ""
        s = start
        for i in range(l):
            ch = random.sample(alphabet, 1)[0]
            s = s.get_nxt(ch)
            seq += ch
        return seq, s.get_acc()

    def random_pos(self, l, start, alphabet):
        while True:
            pos_seq, label = self.random_word(l, start, alphabet)
            if label == True:
                break
        return pos_seq

    def random_neg(self, l, start, alphabet):
        while True:
            neg_seq, label = self.random_word(l, start, alphabet)
            if label == False:
                break
        return neg_seq


class Tomita_1(Tomita):

    def __init__(self):
        # super.__init__()
        self.__re = "1*"
        self.__Sigma = ["0", "1"]
        self.__states = [DFAState(0, True), DFAState(1, False)]
        self.__n_states = 2
        self.__start = self.__states[0]
        self.__states[0].set_nxt("0", self.__states[1])
        self.__states[0].set_nxt("1", self.__states[0])
        self.__states[1].set_nxt("0", self.__states[1])
        self.__states[1].set_nxt("1", self.__states[1])

    def generatePos(self, l, exclude_set, max_iters):
        '''
        With this grammar, only one valid string can be available per length
        :param l: the lenth of string
        :param exclude_set: just to be consistent with others interface
        :param max_iters: just to be consistent with others interface
        :return:
        '''
        pos_seq = self.__Sigma[1] * l
        assert self.classify(pos_seq, self.__start) == True
        return pos_seq

    def generateNeg(self, l, exclude_set, max_iters):
        for i in range(max_iters):
            neg_seq = self.random_neg(l, self.__start, self.__Sigma)
            if neg_seq not in exclude_set:
                break
        return neg_seq

    def get_re(self):

        return copy.deepcopy(self.__re)

    def get_alphabet(self):

        return copy.deepcopy(self.__Sigma)

    def get_start(self):
        return copy.deepcopy(self.__start)


class Tomita_2(Tomita):

    def __init__(self):

        self.__re = "(10)*"
        self.__Sigma = ["0", "1"]
        self.__states = [DFAState(0, True),
                         DFAState(1, False),
                         DFAState(2, True),
                         DFAState(3, False)]
        self.__n_states = 4
        self.__start = self.__states[0]
        self.__states[0].set_nxt("0", self.__states[3])
        self.__states[0].set_nxt("1", self.__states[1])
        self.__states[1].set_nxt("0", self.__states[2])
        self.__states[1].set_nxt("1", self.__states[3])
        self.__states[2].set_nxt("0", self.__states[3])
        self.__states[2].set_nxt("1", self.__states[1])
        self.__states[3].set_nxt("0", self.__states[3])
        self.__states[3].set_nxt("1", self.__states[3])

    def generatePos(self, l, exclude_set, max_iters):
        '''
        With this grammar, only one valid string can be available for a even length
        :param l: the lenth of string
        :param exclude_set: just to be consistent with others interface
        :param max_iters: just to be consistent with others interface
        :return:
        '''
        pos_seq = ""
        if l % 2 == 0:
            pos_l = int(l / 2)
            pos_seq = "10" * pos_l

        if pos_seq != "":
            assert self.classify(pos_seq, self.__start) == True
        return pos_seq

    def generateNeg(self, l, exclude_set, max_iters):
        for i in range(max_iters):
            neg_seq = self.random_neg(l, self.__start, self.__Sigma)
            if neg_seq not in exclude_set:
                break
        if neg_seq != "":
            assert self.classify(neg_seq, self.__start) == False
        return neg_seq

    def get_re(self):

        return copy.deepcopy(self.__re)

    def get_alphabet(self):

        return copy.deepcopy(self.__Sigma)

    def get_start(self):
        return copy.deepcopy(self.__start)


class Tomita_3(Tomita):

    def __init__(self):
        self.__re = "all w without containing an odd number of consecutive 0’s after an odd number of consecutive 1’s"
        self.__Sigma = ["0", "1"]
        self.__states = [DFAState(0, True),
                         DFAState(1, True),
                         DFAState(2, False),
                         DFAState(3, True),
                         DFAState(4, False)]
        self.__n_states = 5
        self.__start = self.__states[0]
        self.__states[0].set_nxt("0", self.__states[0])
        self.__states[0].set_nxt("1", self.__states[1])
        self.__states[1].set_nxt("0", self.__states[2])
        self.__states[1].set_nxt("1", self.__states[0])
        self.__states[2].set_nxt("0", self.__states[3])
        self.__states[2].set_nxt("1", self.__states[4])
        self.__states[3].set_nxt("0", self.__states[2])
        self.__states[3].set_nxt("1", self.__states[3])
        self.__states[4].set_nxt("0", self.__states[4])
        self.__states[4].set_nxt("1", self.__states[4])

    def generatePos(self, l, exclude_set, max_iters):
        for i in range(max_iters):
            if l == 1:
                pos_seq = random.sample(self.__Sigma, 1)[0]
            else:
                pos_seq = self.random_pos(l, self.__start, self.__Sigma)
            if pos_seq not in exclude_set:
                break
        if pos_seq != "":
            assert self.classify(pos_seq, self.__start) == True
        return pos_seq

    def generateNeg(self, l, exclude_set, max_iters):
        for i in range(max_iters):
            if l == 1:
                neg_seq = ""
            else:
                neg_seq = self.random_neg(l, self.__start, self.__Sigma)

            if neg_seq not in exclude_set:
                break

        if neg_seq != "":
            assert self.classify(neg_seq, self.__start) == False
        return neg_seq

    def get_re(self):

        return copy.deepcopy(self.__re)

    def get_alphabet(self):

        return copy.deepcopy(self.__Sigma)

    def get_start(self):
        return copy.deepcopy(self.__start)


class Tomita_4(Tomita):

    def __init__(self):

        self.__re = "((1*)|(01*)|(001*))*"
        self.__Sigma = ["0", "1"]
        self.__states = [DFAState(0, True),
                         DFAState(1, True),
                         DFAState(2, True),
                         DFAState(3, False)]
        self.__n_states = 4
        self.__start = self.__states[0]
        self.__states[0].set_nxt("0", self.__states[1])
        self.__states[0].set_nxt("1", self.__states[0])
        self.__states[1].set_nxt("0", self.__states[2])
        self.__states[1].set_nxt("1", self.__states[0])
        self.__states[2].set_nxt("0", self.__states[3])
        self.__states[2].set_nxt("1", self.__states[0])
        self.__states[3].set_nxt("0", self.__states[3])
        self.__states[3].set_nxt("1", self.__states[3])

    def generatePos(self, l, exclude_set, max_iters):
        for i in range(max_iters):
            pos_seq = self.random_pos(l, self.__start, self.__Sigma)
            if pos_seq not in exclude_set:
                break
        if pos_seq != "":
            assert self.classify(pos_seq, self.__start) == True
        return pos_seq

    def generateNeg(self, l, exclude_set, max_iters):
        for i in range(max_iters):
            if l < 3:
                neg_seq = ""
            else:
                poison = "000"
                rst_len = l - 3
                neg_seq = ""
                for i in range(rst_len):
                    ch = random.sample(self.__Sigma, 1)[0]
                    neg_seq += ch
                # intert the poison
                idx = random.randint(0, rst_len)
                neg_seq = neg_seq[0:idx] + poison + neg_seq[idx:]
            if neg_seq not in exclude_set:
                break

        if neg_seq != "":
            assert self.classify(neg_seq, self.__start) == False

        return neg_seq

    def get_re(self):

        return copy.deepcopy(self.__re)

    def get_alphabet(self):

        return copy.deepcopy(self.__Sigma)

    def get_start(self):
        return copy.deepcopy(self.__start)


class Tomita_5(Tomita):

    def __init__(self):
        super(Tomita_5, self).__init__()
        self.__re = "all w for which the number of 0’s and 1’s are even"
        self.__Sigma = ["0", "1"]
        self.__states = [DFAState(0, True),
                         DFAState(1, False),
                         DFAState(2, False),
                         DFAState(3, False)]
        self.__n_states = 4
        self.__start = self.__states[0]
        self.__states[0].set_nxt("0", self.__states[1])
        self.__states[0].set_nxt("1", self.__states[2])
        self.__states[1].set_nxt("0", self.__states[0])
        self.__states[1].set_nxt("1", self.__states[3])
        self.__states[2].set_nxt("0", self.__states[3])
        self.__states[2].set_nxt("1", self.__states[0])
        self.__states[3].set_nxt("0", self.__states[2])
        self.__states[3].set_nxt("1", self.__states[1])

    def even_decompose(self, l):
        # 1 .list all even numbers that lst l
        # 2. random choose one
        a = random.choice(range(2, l, 2))
        b = l - a
        assert b < l and b % 2 == 0
        return a, l - a

    def generatePos(self, l, exclude_set, max_iters):

        pos_seq = ""
        for i in range(max_iters):
            if l % 2 == 0:
                str_type = random.sample([0, 1], 1)[0]
                if l == 2 or str_type == 0:  # pure, only "0" or "1"
                    pos_seq = random.sample(self.__Sigma, 1)[0] * l
                else:
                    a, b = self.even_decompose(l)
                    pos_seq = ["1"] * a
                    zeros = ["0"] * b
                    pos_seq.extend(zeros)
                    random.shuffle(pos_seq)
                    pos_seq = "".join(pos_seq)
            if pos_seq not in exclude_set:
                break

        if pos_seq != "":
            assert self.classify(pos_seq, self.__start) is True
        return pos_seq

    def generateNeg(self, l, exclude_set, max_iters):
        for i in range(max_iters):
            if l % 2 != 0:
                neg_seq = ""
                for i in range(l):
                    ch = random.sample(self.__Sigma, 1)[0]
                    neg_seq += ch
            else:
                neg_seq = self.random_neg(l, self.__start, self.__Sigma)

            if neg_seq not in exclude_set:
                break

        if neg_seq != "":
            assert not self.classify(neg_seq, self.__start)
        return neg_seq

    def get_re(self):

        return copy.deepcopy(self.__re)

    def get_alphabet(self):

        return copy.deepcopy(self.__Sigma)

    def get_start(self):
        return self.__start

    def get_start(self):
        return copy.deepcopy(self.__start)


class Tomita_6(Tomita):

    def __init__(self):

        self.__re = "all w that the difference between the numbers of 0’s and 1’s is 3n"
        self.__Sigma = ["0", "1"]
        self.__states = [DFAState(0, True),
                         DFAState(1, False),
                         DFAState(2, False)]
        self.__n_states = 3
        self.__start = self.__states[0]
        self.__states[0].set_nxt("0", self.__states[2])
        self.__states[0].set_nxt("1", self.__states[1])
        self.__states[1].set_nxt("0", self.__states[0])
        self.__states[1].set_nxt("1", self.__states[2])
        self.__states[2].set_nxt("0", self.__states[1])
        self.__states[2].set_nxt("1", self.__states[0])

    def get_pos_difference(self, l):
        assert l > 1
        ds = range(0, int(math.floor(l / 3) + 1))
        while True:
            d = random.choice(ds)
            if (3 * d + l) % 2 == 0:
                a = (3 * d + l) / 2
                if a <= l:
                    b = l - a
                    break
        return int(a), int(b)

    def generatePos(self, l, exclude_set, max_iters):
        pos_seq = ""
        for i in range(max_iters):
            if l > 1:
                a, b = self.get_pos_difference(l)
                major_ch = random.sample(["0", "1"], 1)[0]
                if major_ch == "0":
                    zeros = ["0"] * a
                    ones = ["1"] * b
                else:
                    ones = ["1"] * a
                    zeros = ["0"] * b
                ones.extend(zeros)
                random.shuffle(ones)
                pos_seq = "".join(ones)
            if pos_seq not in exclude_set:
                break
        if pos_seq != "":
            assert self.classify(pos_seq, self.__start) is True
        return pos_seq

    def generateNeg(self, l, exclude_set, max_iters):

        for i in range(max_iters):
            neg_seq = self.random_neg(l, self.__start, self.__Sigma)
            if neg_seq not in exclude_set:
                break
        if neg_seq != "":
            assert not self.classify(neg_seq, self.__start)
        return neg_seq

    def get_re(self):

        return copy.deepcopy(self.__re)

    def get_alphabet(self):

        return copy.deepcopy(self.__Sigma)

    def get_start(self):
        return copy.deepcopy(self.__start)


class Tomita_7(Tomita):

    def __init__(self):

        self.__re = "0*1*0*1*"
        self.__Sigma = ["0", "1"]
        self.__states = [DFAState(0, True),
                         DFAState(1, True),
                         DFAState(2, True),
                         DFAState(3, True),
                         DFAState(4, False)]
        self.__n_states = 5
        self.__start = self.__states[0]
        self.__states[0].set_nxt("0", self.__states[0])
        self.__states[0].set_nxt("1", self.__states[1])
        self.__states[1].set_nxt("0", self.__states[2])
        self.__states[1].set_nxt("1", self.__states[1])
        self.__states[2].set_nxt("0", self.__states[2])
        self.__states[2].set_nxt("1", self.__states[3])
        self.__states[3].set_nxt("0", self.__states[4])
        self.__states[3].set_nxt("1", self.__states[3])
        self.__states[4].set_nxt("0", self.__states[4])
        self.__states[4].set_nxt("1", self.__states[4])

    def generatePos(self, l, exclude_set, max_iters):
        pos_seq = ""
        for i in range(max_iters):
            pos_seq = self.random_pos(l, self.__start, self.__Sigma)
            if pos_seq not in exclude_set:
                break

        if pos_seq != "":
            assert self.classify(pos_seq, self.__start) is True
        return pos_seq

    def generateNeg(self, l, exclude_set, max_iters):
        neg_seq = ""
        for i in range(max_iters):
            if l < 4:
                break
            elif l == 4:
                neg_seq = "1010"
            else:
                neg_seq = self.random_neg(l, self.__start, self.__Sigma)
            if neg_seq not in exclude_set:
                break
        if neg_seq != "":
            assert not self.classify(neg_seq, self.__start)
        return neg_seq

    def get_re(self):
        return copy.deepcopy(self.__re)

    def get_alphabet(self):
        return copy.deepcopy(self.__Sigma)

    def get_start(self):
        return copy.deepcopy(self.__start)
