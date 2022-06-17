# coding:utf-8

"""
@Author    :   ShaoCHi
@Date      :   2022/6/2 20:55
@Name      :   feature_engineering.py
@Software  :   PyCharm
"""
import data_preprocessing
from data_preprocessing import *
import numpy as np


def generate_feature_vectors(__allSentences, method):
    if method == 'bag-of-words':
        feature_vectors = []
        for sentence in __allSentences:
            words = word_extraction(sentence)
            bag_vector = np.zeros(len(data_preprocessing.vocabulary), dtype='float32')
            for word in words:
                for i, content in enumerate(data_preprocessing.vocabulary):
                    if content == word:
                        bag_vector[i] += 1
            feature_vectors.append(bag_vector)
        return feature_vectors
    elif method == 'N-gram':
        feature_vectors = []
        for sentence in __allSentences:
            words = word_extraction(sentence)
            single_phrase_words_3_gram = ngrams(words, 3)
            bag_vector = np.zeros(len(data_preprocessing.vocabulary), dtype='float32')
            for word in single_phrase_words_3_gram:
                for i, content in enumerate(data_preprocessing.vocabulary):
                    if content == word:
                        bag_vector[i] += 1
            feature_vectors.append(bag_vector)
        return feature_vectors
