# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 21:48:48 2018
@author: DrLC
"""

import pickle
import numpy as np
import random

from data_factory.tomita import tomita_fa
import torch

class Grammar():

    Tomita1=1
    Tomita2=2
    Tomita3=3
    Tomita4=4
    Tomita5=5
    Tomita6=6
    Tomita7=7

class Generator(object):
    '''
    For each length, we generate the identical number of positive strings and negative strings.
    '''
    def __init__(self,grammarType,len_pool,max_iteries, max_size_per_len, ub_neg_num=20):
        '''
        :param grammarType: tomita grammar
        :param len_pool:list. the list of the string's length.
        :param max_iteries: the max number of iterations trying to randomly generate a desired sample.
        :param max_size_per_len: the max number of samples for each length.
        :param ub_neg_num: the number of negative samples per length under tomita1 and tomita2
        '''
        self.grammarType=grammarType
        self.len_pool=len_pool
        self.max_iteries=max_iteries
        self.max_size_per_len = max_size_per_len
        self.ub_neg_num = ub_neg_num
        self.alphabet=["0","1"]

        if self.grammarType == Grammar.Tomita1:
            self.dfa = tomita_fa.Tomita_1()

        if self.grammarType == Grammar.Tomita2:
            self.dfa = tomita_fa.Tomita_2()

        if self.grammarType == Grammar.Tomita3:
            self.dfa = tomita_fa.Tomita_3()

        if self.grammarType == Grammar.Tomita4:
            self.dfa = tomita_fa.Tomita_4()

        if self.grammarType == Grammar.Tomita5:
            self.dfa = tomita_fa.Tomita_5()

        if self.grammarType == Grammar.Tomita6:
            self.dfa = tomita_fa.Tomita_6()

        if self.grammarType == Grammar.Tomita7:
            self.dfa = tomita_fa.Tomita_7()


    def generate_balance(self):
        data={}
        for l in self.len_pool:
            for i in range(self.max_size_per_len):
                if l==0:
                    data[""]=True
                    break
                if self.dfa.get_re()=="1*" or self.dfa.get_re()=="(10)*":
                    if i >= self.ub_neg_num:
                        break
                pos = self.dfa.generatePos(l,data.keys(),self.max_iteries)
                neg=self.dfa.generateNeg(l,data.keys(),self.max_iteries)
                if pos not in data.keys() and len(pos)>0:
                    data[pos]=True
                if neg not in data.keys() and len(neg)>0:
                    data[neg]=False
        return data


    def generate_random(self,data_size):
        data={}
        random.seed(20190101)
        for i in range(data_size):
            while True:
                l = random.choice(self.len_pool)
                seq,label=self.dfa.random_word(l,start=self.dfa.get_start(),alphabet=self.dfa.get_alphabet())
                if seq not in data.keys():
                    if label is True:
                        data[seq]=True
                    else:
                        data[seq]=False
                    break
        return data

    def save_data(self,save_path,data):
        '''
        save data into a pickle file: {"data":{"str":lable},"num_pos":int,"num_neg":int}
        :param save_path:
        :param pos_list:
        :param neg_list:
        :return:
        '''
        with open(save_path,"wb") as f:
            num_pos=len([seq for seq in data.keys() if data[seq]==True])
            num_neg=len([seq for seq in data.keys() if data[seq]==False])
            assert num_neg+num_pos==len(data)
            print("{}({})".format(len(data), num_pos))
            pickle.dump({"data":data,
                         "num_pos":num_pos,
                         "num_neg":num_neg},f)

class TomitaDataProcessor(object):
    def sequence2tensor(self, sequence, input_tensor_dim=3, is_single_step=False):
        '''
        use tensor([1,0]) to denote character "0" and tensor([0,1]) to denote character "1"
        :param sequences: e.g.,"100010100"
        :param input_tensor_dim: the length of one-hot. since we take "",i.e.,empty string, into account, so the size of
                                 alphabet is 3.
        :param is_batch:
        :return:
        '''

        def make_one_hot(ch):
            if ch == "":
                vector = torch.tensor([1, 0, 0])
            elif ch == "0":
                vector = torch.tensor([0, 1, 0])
            else:
                vector = torch.tensor([0, 0, 1])
            return vector

        len_seq = 1 if sequence=="" or is_single_step else len(sequence)+1 # add the empty string
        sequence_tensor = torch.zeros(1, len_seq, input_tensor_dim)

        if len_seq == 1:
            vector = make_one_hot(sequence)
            sequence_tensor[0][0] = vector
            return sequence_tensor
        else:
            sequence_tensor[0][0] = make_one_hot("") # add the empty string
            for li,ch in enumerate(sequence):
                sequence_tensor[0][li+1] = make_one_hot(ch)
        return sequence_tensor


    def load_data(self,data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        return data["data"]


def make_training_set():
    for grammar in [Grammar.Tomita1,Grammar.Tomita2,Grammar.Tomita3, Grammar.Tomita4, Grammar.Tomita5, Grammar.Tomita6, Grammar.Tomita7]:
        print("Generating Tomita{}....".format(grammar))
        save_path="../../data/tomita/training/tomita"+str(grammar)+".pkl"
        len_pool = range(14)
        if grammar == Grammar.Tomita6:
            # 0-13,15-20
            len_pool.extend(range(15,21))
        else:
            # 0-13,16,19,22
            len_pool.extend(range(16,23,3))

        max_iteries = 100
        max_size_per_len = 150 # 150*2 =300
        ub_neg_num = 50

        g=Generator(grammar,len_pool,max_iteries,max_size_per_len,ub_neg_num)
        data=g.generate_balance()
        g.save_data(save_path,data)


def make_test_set():
    for grammar in [Grammar.Tomita1,Grammar.Tomita2,Grammar.Tomita3, Grammar.Tomita4, Grammar.Tomita5, Grammar.Tomita6, Grammar.Tomita7]:
        print("Generating Tomita{}....".format(grammar))
        save_path="../../data/tomita/test/tomita"+str(grammar)+".pkl"
        len_pool = range(1,29,3)

        max_iteries = 100
        max_size_per_len = 150 # 150*2 =300
        ub_neg_num = 50
        test_data_size = 1000

        g=Generator(grammar,len_pool,max_iteries,max_size_per_len,ub_neg_num)
        data=g.generate_random(test_data_size)
        g.save_data(save_path,data)


if __name__=="__main__":
    make_test_set()
    # make_training_set()



    # t=tomita.Tomita_5()
    # print(t.classify("00",t.get_start()))

