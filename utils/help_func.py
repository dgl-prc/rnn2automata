import sys
sys.path.append("../")
import os
import torch
import numpy as np
import re
import pickle


def save_readme(parent_path, content):
    with open(os.path.join(parent_path, "README"), "w") as f:
        f.writelines(content)

def clean_data_for_look(text):
    REPLACE_BY_SPACE_RE = re.compile('[/{}\[\]\|@#*]')
    REPLACE_BY_ANONYMOUS = re.compile('((x|X){2,}\s*)+')  # replace the XXXX by XUNATIE
    REDUCE_REPEATED_DOTS_RE = re.compile('(\.){2,}')  # replace the repeated '.' and '?'
    REDUCE_REPEATED_EXCLAMATION_RE = re.compile('!{2,}')
    REDUCE_REPEATED_QUESTION_RE = re.compile('\?{2,}')
    REDUCE_REPEATED_SPACE_RE = re.compile('\s{2,}')  # remove repeated space
    REMOVE_HTML_RE = re.compile('<.*?>')  # remove html marks.

    text = REMOVE_HTML_RE.sub(' ', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = REPLACE_BY_ANONYMOUS.sub('ZJU ', text)
    text = REDUCE_REPEATED_DOTS_RE.sub('. ', text)
    REDUCE_REPEATED_QUESTION_RE.sub("? ", text)
    text = REDUCE_REPEATED_EXCLAMATION_RE.sub('! ', text)

    ###################################
    # always should be the last
    ###################################
    text = REDUCE_REPEATED_SPACE_RE.sub(' ', text)
    return text.strip()

def save_model(model, train_acc, test_acc, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_acc = "{0:.4f}".format(train_acc)
    test_acc = "{0:.4f}".format(test_acc)
    save_file = os.path.join(save_path, 'train_acc-' + train_acc + '-test_acc-' + test_acc + '.pkl')
    torch.save(model.cpu().state_dict(), save_file)

def make_check_point_folder(task_name,dataset, modelType):
    check_point_folder = os.path.join("./tmp",task_name, dataset, modelType)
    if not os.path.exists(check_point_folder):
        os.makedirs(check_point_folder)
    return check_point_folder

def converPickle2python2(pickle_file):
    with open(pickle_file, "rb") as f:
        obj = pickle.load(f)

    file_name, file_exten = os.path.splitext(pickle_file)
    new_file_name = "".join([file_name, "protocol2", file_exten])
    with open(new_file_name, "wb") as f:
        pickle.dump(obj, f, protocol=2)

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        pkl_obj = pickle.load(f)
    return pkl_obj

def save_pickle(file_path, obj, protocol=3):
    parent_path = os.path.split(file_path)[0]
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)

def save_adv_text(file_path, ori_labels, adv_lables, adv_sentences):
    paren_path = os.path.split(file_path)[0]
    if not os.path.exists(paren_path):
        os.mkdir(paren_path)
    with open(file_path, "wb") as f:
        pickle.dump({"original_labels": ori_labels,
                     "adv_labels": adv_lables,
                     "adv_sentences": adv_sentences}, f)

def load_adv_text(file_path, shuffle=True):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        ori_labels = data["original_labels"]
        adv_lables = data["adv_labels"]
        adv_sentences = data["adv_sentences"]
    if shuffle:
        idx = [i for i in range(len(ori_labels))]
        np.random.shuffle(idx)
        ori_labels = [ori_labels[i] for i in idx]
        adv_lables = [adv_lables[i] for i in idx]
        adv_sentences = [adv_sentences[i] for i in idx]
    return ori_labels, adv_lables, adv_sentences


