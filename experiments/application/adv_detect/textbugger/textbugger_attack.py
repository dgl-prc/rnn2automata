import spacy
import copy
import random
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import gensim
from utils.constant import *
from sklearn.metrics.pairwise import cosine_similarity


class TextBugger(object):
    '''
    The classifier should implement the following interface.
    def get_label(self, sent)
    def get_probs(self, sent)
    '''
    DELETE_C = "DELC"
    INSERT = "INSERT"
    SWAP_C = "SWAPC"
    SUB_C = "SUBC"
    SUB_W = "SUBW"

    def __init__(self, classifier, word2vec_model, bug_mode):
        self.nlp = spacy.load("en")
        self.classifier = classifier
        self.topn = 5
        self.sim_epsilon = 0.8
        self.bug_mode = bug_mode
        if isinstance(word2vec_model, str):
            self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model, binary=False)
        else:
            self.word2vec = word2vec_model

    def document2sents(self, input_text):
        assert isinstance(input_text, str)
        document = self.nlp(input_text)
        sentence_list = []
        for sentence in document.sents:
            sentence_list.append(sentence.text.split(" "))
        return sentence_list

    def sentsList2wordsList(self, sents_list):
        sents = []
        for item in sents_list:
            sents += item
        return sents

    def sort_sents(self, sent_list, ori_label):
        sent_scores = []
        MINIM_SCORE = -100
        for sentence in sent_list:
            probs = self.classifier.get_probs(sentence)
            label = np.argmax(probs)
            prob = probs[label]
            if label != ori_label:
                sent_scores.append(MINIM_SCORE)
            else:
                sent_scores.append(prob)
        sorted_list = np.argsort(sent_scores)[::-1]

        return [idx for idx in sorted_list if sent_scores[idx] != MINIM_SCORE]

    def sort_words(self, sentence):
        if len(sentence) == 1:
            sorted_words_idx = [0]
        else:
            word_scores = []
            sent_probs = self.classifier.get_probs(sentence)
            sent_label = np.argmax(sent_probs)
            sent_prob = sent_probs[sent_label]
            for w_idx in range(len(sentence)):
                new_sent = copy.deepcopy(sentence)
                new_sent.pop(w_idx)
                new_prob = self.classifier.get_probs(new_sent)[sent_label]
                word_scores.append(sent_prob - new_prob)
            sorted_words_idx = np.argsort(word_scores)[::-1]
        return sorted_words_idx

    def replace_word(self, new_word, sent_idx, w_idx, sent_list):
        '''
        replace the w_idx of sent_idx in sent_list with new_word
        :param word:
        :param sent_idx:
        :param w_idx:
        :param sent_list:
        :return:
        '''
        new_sent_list = copy.deepcopy(sent_list)
        new_sent_list[sent_idx][w_idx] = new_word
        return new_sent_list

    def insert_space(self):
        """
         Insert a space into the word. Generally, words are segmented by spaces in English.
         Considering the usability of text, we apply this method only when the length of the
         word is shorter than 6 characters since long words might be split into two legitimate words
        :return:
        """
        pass

    def delete_character(self, sent_idx, w_idx, sent_list):
        '''
         Delete a random character of the word except for the first and the last character
        :return:
        '''
        target_word = sent_list[sent_idx][w_idx]
        new_word = [c for c in target_word]
        if len(target_word) < 3:
            return sent_list
        c_idxs = [i for i in range(len(target_word))]
        i = random.choice(c_idxs[1:-1])
        new_word.pop(i)
        new_word = "".join(new_word)
        new_sent = copy.deepcopy(sent_list)
        new_sent[sent_idx][w_idx] = new_word
        return new_sent

    def swap_characters(self, sent_idx, w_idx, sent_list):
        """
         Swap random two adjacent letters in the word but do not alter the first or last
         letter.
         For this reason, this method is only applied to words longer than 4 letters.
        :return:
        """
        target_word = sent_list[sent_idx][w_idx]
        if len(target_word) < 4:
            return sent_list

        new_word = [c for c in target_word]
        c_idxs = [i for i in range(len(target_word))]
        i, j = random.sample(c_idxs[1:-1], 2)
        ci, cj = new_word[i], new_word[j]
        new_word[i], new_word[j] = cj, ci
        new_sent = copy.deepcopy(sent_list)
        new_sent[sent_idx][w_idx] = new_word
        return new_sent

    def substitute_character(self, sent_idx, w_idx, sent_list):
        '''
         Replace characters with visually similar characters (e.g., replacing “o”
         with “0”, “l” with “1”, “a” with “@”) or adjacent characters in
         the keyboard (e.g., replacing “m” with “n”)
        :return:
        '''
        pass

    def selectBug(self, sent_idx, w_idx, sent_list, label, prob):
        '''
        Substitute-W
        :return:
        '''
        target_word = sent_list[sent_idx][w_idx]
        if target_word not in self.word2vec.vocab:
            return sent_list
        if self.bug_mode == TextBugger.SUB_W:
            words_list = self.word2vec.most_similar(positive=[target_word], topn=self.topn)
            bugs = [item[0] for item in words_list]
            max_score = 0
            best_newSentList = []
            for bug in bugs:
                new_sent_list = self.replace_word(bug, sent_idx, w_idx, sent_list)
                new_probs = self.classifier.get_probs(self.sentsList2wordsList(new_sent_list))
                new_prob = new_probs[label]
                score = prob - new_prob
                if score > max_score:
                    best_newSentList = new_sent_list
                    max_score = score
            if max_score == 0:
                new_sent = sent_list
            else:
                new_sent = best_newSentList
        elif self.bug_mode == TextBugger.DELETE_C:
            new_sent = self.delete_character(sent_idx, w_idx, sent_list)
        elif self.bug_mode == TextBugger.SWAP_C:
            new_sent = self.swap_characters(sent_idx, w_idx, sent_list)
        else:
            new_sent = self.substitute_character(sent_idx, w_idx, sent_list)

        return new_sent

    def similarity(self, x1, x2):
        assert isinstance(x1, list)
        assert isinstance(x2, list)
        with tf.Graph().as_default():
            self.sentence_encoder = hub.Module(SENTENCE_ENCODER)
            with tf.Session() as tf_sess:
                tf_sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
                x1_x2 = tf_sess.run(self.sentence_encoder([" ".join(x1), " ".join(x2)]))
        sm = cosine_similarity(X=x1_x2)
        return sm[0][1]

    def attack(self, input_text):
        ori_label = self.classifier.get_label(input_text)
        prob = self.classifier.get_probs(input_text)[ori_label]
        if isinstance(input_text, list):
            sent_list = self.document2sents(" ".join(input_text))
        else:
            sent_list = self.document2sents(input_text)
        sorted_list = self.sort_sents(sent_list, ori_label)
        newSentsList = copy.deepcopy(sent_list)
        for sent_idx in sorted_list:
            sentence = sent_list[sent_idx]
            sorted_words_idx = self.sort_words(sentence)
            for w_idx in sorted_words_idx:
                newSentsList = self.selectBug(sent_idx, w_idx, newSentsList, ori_label, prob)
                newSent = self.sentsList2wordsList(newSentsList)
                newLabel = self.classifier.get_label(newSent)
                if newLabel != ori_label:
                    sim = self.similarity(input_text, newSent)
                    if sim > self.sim_epsilon:
                        return newSent, newLabel
        return None, -1
