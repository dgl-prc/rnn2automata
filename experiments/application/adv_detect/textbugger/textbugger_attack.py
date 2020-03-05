import numpy as np
import spacy
import copy
import tensorflow_hub as hub
import tensorflow as tf
import gensim
from utils.constant import *
from utils.model_data import *
from sklearn.metrics.pairwise import cosine_similarity

class TextBugger(object):
    '''
    The classifier should implement the following interface.
    def get_label(self, sent)
    def get_probs(self, sent)
    '''
    def __init__(self, classifier, word2vec_model):
        self.nlp = spacy.load("en")
        self.classifier = classifier
        self.topn = 5
        self.sim_epsilon = 0.8
        if isinstance(word2vec_model,str):
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

    def selectBug(self, sent_idx, w_idx, sent_list, label, prob):
        '''
        Substitute-W
        :return:
        '''
        target_word = sent_list[sent_idx][w_idx]
        if target_word not in self.word2vec.vocab:
            return sent_list
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
            return sent_list
        return best_newSentList

    def similarity(self,x1, x2):
        assert isinstance(x1,list)
        assert isinstance(x2,list)
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
                    sim = self.similarity(input_text,newSent)
                    if sim > self.sim_epsilon:
                        return newSent, newLabel
        return None, -1





