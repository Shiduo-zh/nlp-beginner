# coding:utf-8

"""
@Author    :   ShaoCHi
@Date      :   2022/6/2 20:32
@Name      :   data_preprocessing.py
@Software  :   PyCharm
"""
import re
from nltk.util import ngrams

vocabulary = []
r = "[^0-9A-Za-z\u4e00-\u9fa5 ]"


def word_extraction(__sentence):
    words = __sentence.split()
    extraction_words = [word.lower() for word in words]
    return extraction_words


# tokenization through word extraction
def tokenize(__sentences, __method):
    if __method == 'bag-of-words':
        vocab = []
        for sentence in __sentences:
            single_phrase_words = word_extraction(sentence)
            vocab.extend(single_phrase_words)
        vocab = sorted(list(set(vocab)))
        return vocab
    elif __method == 'N-gram':
        vocab = []
        for sentence in __sentences:
            single_phrase_words = word_extraction(sentence)
            # the value of n is hyperparameter
            single_phrase_words_3_gram = ngrams(single_phrase_words, 3)
            vocab.extend(single_phrase_words_3_gram)
        vocab = sorted(list(set(vocab)))
        return vocab


# get the vocabulary of the data
def data_preprocess(data, method):
    global vocabulary
    data["Phrase"] = data.apply(lambda row: re.sub(r, '', row['Phrase']), axis=1)
    vocabulary = tokenize(data["Phrase"], method)
    return data
