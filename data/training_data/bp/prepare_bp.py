"""
This BP generation code is from "https://github.com/tech-srl/lstar_extraction/Specific_Language_Generation.py"
"""
import random
import string
import numpy as np
from random import shuffle
from data.training_data.artificial_data_utils import n_words_of_length
from utils.help_func import save_pickle
from utils.constant import *
from data.text_utils import set_data, make_wv_matrix

bp_other_letters = string.ascii_lowercase  # probably avoid putting '$' in here because that's my dummy letter somewhere in the network (todo: something more general)
alphabet_bp = "()" + bp_other_letters


def make_train_set_for_target(target, alphabet, lengths=None, max_train_samples_per_length=300,
                              search_size_per_length=1000, provided_examples=None):
    train_set = {}
    if None == provided_examples:
        provided_examples = []
    if None == lengths:
        lengths = list(range(15)) + [15, 20, 25, 30]
    for l in lengths:
        samples = [w for w in provided_examples if len(w) == l]
        samples += n_words_of_length(search_size_per_length, l, alphabet)
        pos = [w for w in samples if target(w)]
        neg = [w for w in samples if not target(w)]
        pos = pos[:int(max_train_samples_per_length / 2)]
        neg = neg[:int(max_train_samples_per_length / 2)]
        minority = min(len(pos), len(neg))
        pos = pos[:minority + 20]
        neg = neg[:minority + 20]
        train_set.update({w: True for w in pos})
        train_set.update({w: False for w in neg})
    print("made train set of size:", len(train_set), ", of which positive examples:",
          len([w for w in train_set if train_set[w] == True]))
    num_pos = len([w for w in train_set.keys() if train_set[w] == True])
    return train_set, num_pos, len(train_set) - 1


def make_similar(w, alphabet):
    new = list(w)
    indexes = list(range(len(new)))
    # switch characters
    num_switches = random.choice(range(3))
    shuffle(indexes)
    indexes_to_switch = indexes[:num_switches]
    for i in indexes_to_switch:
        new[i] = random.choice(alphabet)
    # insert characters
    num_inserts = random.choice(range(3))
    indexes = indexes + [len(new)]
    indexes_to_insert = indexes[:num_inserts]
    for i in indexes_to_insert:
        new = new[:i] + [random.choice(alphabet)] + new[i:]
    num_changes = num_switches + num_inserts
    # duplicate letters
    while ((num_changes == 0) or (random.choice(range(3)) == 0)) and len(new) > 0:
        index = random.choice(range(len(new)))
        new = new[:index + 1] + new[index:]
        num_changes += 1
    # omissions
    while ((num_changes == 0) or random.choice(range(3)) == 0) and len(new) > 0:
        index = random.choice(range(len(new)))
        new = new[:index] + new[index + 1:]
        num_changes += 1
    return ''.join(new)


def balanced_parantheses(w):
    open_counter = 0
    while len(w) > 0:
        c = w[0]
        w = w[1:]
        if c == "(":
            open_counter += 1
        elif c == ")":
            open_counter -= 1
            if open_counter < 0:
                return False
    return open_counter == 0


def random_balanced_word(start_closing):
    count = 0
    word = ""
    while len(word) < start_closing:
        paran = (random.choice(range(3)) == 0)
        next_letter = random.choice("()") if paran else random.choice(bp_other_letters)
        if next_letter == ")" and count <= 0:
            continue
        word += next_letter
        if next_letter == "(":
            count += 1
        if next_letter == ")":
            count -= 1
    while True:
        paran = (random.choice(range(3)) == 0)
        next_letter = random.choice(")") if paran else random.choice(bp_other_letters)
        if next_letter == ")":
            count -= 1
            if count < 0:
                break
        word += next_letter
    return word


def n_balanced_words_around_lengths(n, short, longg):
    words = set()
    while len(words) < n:
        for l in range(short, longg):
            words.add(random_balanced_word(l))
    #     print('\n'.join(sorted(list(words),key=len)))
    return words


def get_balanced_parantheses_train_set(n, short, longg, lengths=None, max_train_samples_per_length=300,
                                       search_size_per_length=200):  # eg 15000, 2, 30
    balanced_words = list(n_balanced_words_around_lengths(n, short, longg))
    almost_balanced = [make_similar(w, alphabet_bp) for w in balanced_words][:int(2 * n / 3)]
    less_balanced = [make_similar(w, alphabet_bp) for w in almost_balanced]
    barely_balanced = [make_similar(w, alphabet_bp) for w in less_balanced]

    all_words = balanced_words + almost_balanced + less_balanced + barely_balanced
    return make_train_set_for_target(balanced_parantheses, alphabet_bp, lengths=lengths, \
                                     max_train_samples_per_length=max_train_samples_per_length, \
                                     search_size_per_length=search_size_per_length, \
                                     provided_examples=all_words)


def make_wv_matrix_bp(data):
    index_map = {key: i + 1 for i, key in enumerate(string.ascii_lowercase)}
    index_map[''] = 0
    index_map["("] = 27
    index_map[")"] = 28
    wv_matrix = []
    input_dim = 29  # alphabet_size
    for i in range(len(data["vocab"])):
        word = data["idx_to_word"][i]
        vector = [0.] * input_dim
        vector[index_map[word]] = 1.
        wv_matrix.append(vector)
    wv_matrix = np.array(wv_matrix)
    return wv_matrix


def divide_bp():
    _n = 44600
    _min_dep = 2
    _max_dep = 11
    bp_data, num_pos, num_neg = get_balanced_parantheses_train_set(_n, _min_dep, _max_dep, lengths=None,
                                                                   max_train_samples_per_length=300,
                                                                   search_size_per_length=200)
    X = []
    Y = []
    for word in bp_data:
        x = [w for w in word] if word != "" else ['']
        X.append(x)
        Y.append(int(bp_data[word]))
    data = set_data(X, Y)
    wv_matrix = make_wv_matrix_bp(data)
    save_path = get_path(DataPath.BP.PROCESSED_DATA)
    save_wv_matrix_path = get_path(DataPath.BP.WV_MATRIX)
    print("train: {}({}), test:{}({})".format(len(data["train_y"]), sum(data["train_y"]), len(data["test_y"]),
                                              sum(data["test_y"])))
    save_pickle(save_path, data)
    save_pickle(save_wv_matrix_path, wv_matrix)
    print("saved in {}".format(save_path))
