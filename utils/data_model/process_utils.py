# import numpy as np
# import re
# import torch
#
#
# def clean_data_for_look(text):
#     REPLACE_BY_SPACE_RE = re.compile('[/{}\[\]\|@#*]')
#     REPLACE_BY_ANONYMOUS = re.compile('((x|X){2,}\s*)+')  # replace the XXXX by XUNATIE
#     REDUCE_REPEATED_DOTS_RE = re.compile('(\.){2,}')  # replace the repeated '.' and '?'
#     REDUCE_REPEATED_EXCLAMATION_RE = re.compile('!{2,}')
#     REDUCE_REPEATED_QUESTION_RE = re.compile('\?{2,}')
#     REDUCE_REPEATED_SPACE_RE = re.compile('\s{2,}')  # remove repeated space
#     REMOVE_HTML_RE = re.compile('<.*?>')  # remove html marks.
#
#     text = REMOVE_HTML_RE.sub(' ', text)
#     text = REPLACE_BY_SPACE_RE.sub(' ', text)
#     text = REPLACE_BY_ANONYMOUS.sub('ZJU ', text)
#     text = REDUCE_REPEATED_DOTS_RE.sub('. ', text)
#     REDUCE_REPEATED_QUESTION_RE.sub("? ", text)
#     text = REDUCE_REPEATED_EXCLAMATION_RE.sub('! ', text)
#
#     ###################################
#     # always should be the last
#     ###################################
#     text = REDUCE_REPEATED_SPACE_RE.sub(' ', text)
#     return text.strip()
#
#
# def make_wv_matrix(data, word_vectors):
#     wv_matrix = []
#     for i in range(len(data["vocab"])):
#         word = data["idx_to_word"][i]
#         if word in word_vectors.vocab:
#             wv_matrix.append(word_vectors.word_vec(word))
#         else:
#             wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
#     # one for UNK and one for zero padding
#     wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
#     wv_matrix.append(np.zeros(300).astype("float32"))
#     wv_matrix = np.array(wv_matrix)
#     return wv_matrix
#
#
# def set_data(x, y, test_idx):
#     data = {}
#     data["train_x"], data["train_y"] = x[:test_idx], y[:test_idx]
#     data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]
#     data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["test_x"] for w in sent])))
#     data["classes"] = sorted(list(set(data["train_y"])))
#     data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
#     data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}
#     return data
#
#
#
# class RealDataProcessor(object):
#     def __init__(self,device, WORD2IDX, INPUT_SIZE, WV_MATRIX):
#         self.device = device
#         self.WORD2IDX = WORD2IDX
#         self.INPUT_SIZE = INPUT_SIZE
#         self.WV_MATRIX = WV_MATRIX
#
#     def sequence2tensor(self,sequence,input_dim):
#         return sent2tensor(sequence, self.device, self.WORD2IDX, self.INPUT_SIZE, self.WV_MATRIX)
#
#     def sequence_purifier(self,sequence):
#         '''
#         since the new approach
#         '''
#         return sequence
#
