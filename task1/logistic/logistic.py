import pandas as pd
from scipy.sparse import hstack
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from plot_matrix import plot_matrix
import numpy as np

# 读取数据
data_train = pd.read_csv('../data/train.tsv', sep='\t')
data_test = pd.read_csv('../data/test.tsv', sep='\t')
x_train = data_train['Phrase']
y_train = data_train['Sentiment']
x_test = data_train['Phrase']
y_test = data_train['Sentiment']
# 划分数据集 train:cv:test=6:2:2
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# 文本向量化

vec = CountVectorizer(stop_words='english')
x_train_vec = vec.fit_transform(x_train)
x_test_vec = vec.transform(x_test)
print('特征数量：', len(vec.get_feature_names()))

idf_vec = TfidfVectorizer(stop_words='english')
x_train_idf_vec = idf_vec.fit_transform(x_train)
x_test_idf_vec = idf_vec.transform(x_test)
print('特征数量：', len(idf_vec.get_feature_names()))

tfidf_ngram_vec = TfidfVectorizer(stop_words='english', ngram_range=(2, 3))
x_traintfidf_ngram_vec = tfidf_ngram_vec.fit_transform(x_train)
x_testtfidf_ngram_vecc = tfidf_ngram_vec.transform(x_test)
print('特征数量：', len(tfidf_ngram_vec.get_feature_names()))

train_features = x_train_vec
test_features = x_test_vec
train_features = hstack([x_train_vec, x_train_idf_vec, x_traintfidf_ngram_vec])
test_features = hstack([x_test_vec, x_test_idf_vec, x_testtfidf_ngram_vecc])
print(train_features.shape)  # 特征的最终维度

lr = LogisticRegression(solver='sag')  # 随机梯度下降
lr.fit(train_features, y_train)
y_predict = lr.predict(test_features)
print(x_train_vec.shape)
print(y_predict)
print('accuracy:', metrics.accuracy_score(y_test, y_predict))
print("precision score", precision_score(y_test, y_predict, average='weighted'))
print("recall score", recall_score(y_test, y_predict, average='weighted'))
print("F1 score", f1_score(y_test, y_predict, average='weighted'))

plot_matrix(y_test, y_predict, [0, 1, 2, 3, 4], title='confusion_matrix_svc',
            axis_labels=['negative', 'somewhat negative', 'neutral', 'somewhat positive', 'positive'])
