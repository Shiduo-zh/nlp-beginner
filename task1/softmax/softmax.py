import random

import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from plot_matrix import plot_matrix


class SoftmaxRegression:
    def __init__(self, K, X, alpha=0.1, lamda=0.0001, tol=1e-4):
        self.num_of_class = K
        self.n = X.shape[0]  # 数据的个数
        self.m = X.shape[1]  # 数据维度,即有多少种数据
        self.weights = None  # 模型权重 shape(类别数，数据维度) 一行代表1个类别的各个维度的权重
        self.alpha = alpha  # 学习率
        # 正则项参数
        self.lamda = lamda

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def fit(self, x_train, y_train, method="BGD", max_itr=100, alpha=0.1, lamda=0,
            batch_size=64):  # shuffle gradient decent
        # 先随机各个类别的数据的权重
        self.weights = np.random.randn(self.num_of_class, self.m)
        self.alpha = alpha

        loss_history = []

        print("-----Training-----")
        for itr in range(max_itr):
            print('iteration:%d' % itr)

            # shuffle gradient decent
            if method == 'SGD':
                rand_index = np.arange(len(x_train))
                np.random.shuffle(rand_index)
                k = random.randint(0, self.n - 1)
                x = x_train[k].reshape(-1, 1)  # 取出每一个样本的特征，并转置，设置为特征数n*1的向量形式
                y = np.zeros((self.num_of_class, 1))
                y[y_train.iloc[k]] = 1  # 将y设置为one-hot的向量形式
                h_y = self.softmax(np.dot(self.weights, x))  # 预测值
                # self.weights = self.weights - alpha * (np.dot((h_y - y), x.T))  # 随机梯度下降更新参数
                self.weights = self.weights - self.alpha * (
                    np.dot((h_y - y), x.T)) + self.alpha * self.lamda * self.weights  # 加入了正则项,lamda默认为0，相当于没有正则化
            # batch gradient decent
            elif method == 'BGD':
                err = np.zeros((self.num_of_class, self.m))
                for i in range(self.n):
                    x = x_train[i].reshape(-1, 1)  # 取出每一个样本的特征，并转置，设置为特征数n*1的向量形式
                    y = np.zeros((self.num_of_class, 1))
                    y[y_train.iloc[i]] = 1  # 将y设置为one-hot的向量形式
                    h_y = self.softmax(np.dot(self.weights, x))  # 预测值
                    singleErr = np.dot((h_y - y), x.T)
                    err += singleErr
                err = err / self.n
                regular = (self.alpha * self.lamda * self.weights) / self.n  # 正则项
                self.weights = self.weights - self.alpha * err + regular
            # mini-batch gradient decent
            elif method == 'mini':
                err = np.zeros((self.num_of_class, self.m))
                for j in range(batch_size):  # 随机抽batch_size次
                    k = random.randint(0, self.n - 1)
                    x = x_train[k].reshape(-1, 1)  # 取出每一个样本的特征，并转置，设置为特征数n*1的向量形式
                    y = np.zeros((self.num_of_class, 1))
                    y[y_train.iloc[k]] = 1  # 将y设置为one-hot的向量形式
                    h_y = self.softmax(np.dot(self.weights, x))  # 预测值
                    singleErr = np.dot((h_y - y), x.T)
                    err += singleErr
                err = err / batch_size
                regular = (self.alpha * self.lamda * self.weights) / batch_size  # 正则项
                self.weights = self.weights - self.alpha * err + regular
                err = np.zeros((self.num_of_class, self.m))

        print('-----Train Finished-----')
        return self.weights

    def predict(self, x_test):
        y_predict = []
        for i in range(len(x_test)):
            x = x_test[i]
            y = np.argmax(np.dot(self.weights, x))
            y_predict.append(y)

        return y_predict

    def test(self, y_test, y_predict):
        print('-----Testing-----')
        accuracy = 0
        for i in range(len(y_test)):
            if y_predict[i] == y_test.iloc[i]:
                accuracy += 1

        accuracy = accuracy / len(y_test)
        print('-----Test Finished-----')
        return accuracy


if __name__ == '__main__':
    # 读取数据
    data_train = pd.read_csv('../data/train.tsv', sep='\t')
    data_test = pd.read_csv('../data/test.tsv', sep='\t')
    x_train = data_train['Phrase'][:8000]
    y_train = data_train['Sentiment'][:8000]
    x_test = data_train['Phrase'][:8000]
    y_test = data_train['Sentiment'][:8000]

    # 文本向量化
    vec = CountVectorizer(stop_words='english')
    x_train_vec = vec.fit_transform(x_train)
    x_test_vec = vec.transform(x_test)

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
    # train_features = hstack([x_train_vec, x_train_idf_vec, x_traintfidf_ngram_vec])
    # test_features = hstack([x_test_vec, x_test_idf_vec, x_testtfidf_ngram_vecc])
    x_train_arr = train_features.toarray()
    x_test_arr = test_features.toarray()
    method='BGD'
    learningRate=16000


    # 实例化softmaxregression
    sl0 = SoftmaxRegression(5, train_features)
    sl0.fit(x_train_arr, y_train, method,learningRate , 0.01)
    y_predict = sl0.predict(x_test_arr)
    accuracy = sl0.test(y_test, y_predict)
    print(accuracy)
    print('accuracy:', metrics.accuracy_score(y_test, y_predict))
    print("precision score", precision_score(y_test, y_predict, average='weighted'))
    print("recall score", recall_score(y_test, y_predict, average='weighted'))
    print("F1 score", f1_score(y_test, y_predict, average='weighted'))
    plot_matrix(y_test, y_predict, [0, 1, 2, 3, 4], title='confusion_matrix_svc',
                axis_labels=['negative', 'somewhat negative', 'neutral', 'somewhat positive', 'positive'])


    sl1 = SoftmaxRegression(5, train_features)
    sl1.fit(x_train_arr, y_train, method,learningRate , 0.1)
    y_predict = sl1.predict(x_test_arr)
    accuracy = sl1.test(y_test, y_predict)
    print(accuracy)
    print('accuracy:', metrics.accuracy_score(y_test, y_predict))
    print("precision score", precision_score(y_test, y_predict, average='weighted'))
    print("recall score", recall_score(y_test, y_predict, average='weighted'))
    print("F1 score", f1_score(y_test, y_predict, average='weighted'))
    plot_matrix(y_test, y_predict, [0, 1, 2, 3, 4], title='confusion_matrix_svc',
                axis_labels=['negative', 'somewhat negative', 'neutral', 'somewhat positive', 'positive'])

    sl2 = SoftmaxRegression(5, train_features)
    sl2.fit(x_train_arr, y_train, method,learningRate , 1)
    y_predict = sl2.predict(x_test_arr)
    accuracy = sl2.test(y_test, y_predict)
    print(accuracy)
    print('accuracy:', metrics.accuracy_score(y_test, y_predict))
    print("precision score", precision_score(y_test, y_predict, average='weighted'))
    print("recall score", recall_score(y_test, y_predict, average='weighted'))
    print("F1 score", f1_score(y_test, y_predict, average='weighted'))
    plot_matrix(y_test, y_predict, [0, 1, 2, 3, 4], title='confusion_matrix_svc',
                axis_labels=['negative', 'somewhat negative', 'neutral', 'somewhat positive', 'positive'])

    sl3 = SoftmaxRegression(5, train_features)
    sl3.fit(x_train_arr, y_train, method,learningRate , 10)
    y_predict = sl3.predict(x_test_arr)
    accuracy = sl3.test(y_test, y_predict)
    print(accuracy)
    print('accuracy:', metrics.accuracy_score(y_test, y_predict))
    print("precision score", precision_score(y_test, y_predict, average='weighted'))
    print("recall score", recall_score(y_test, y_predict, average='weighted'))
    print("F1 score", f1_score(y_test, y_predict, average='weighted'))
    plot_matrix(y_test, y_predict, [0, 1, 2, 3, 4], title='confusion_matrix_svc',
                axis_labels=['negative', 'somewhat negative', 'neutral', 'somewhat positive', 'positive'])

    sl4 = SoftmaxRegression(5, train_features)
    sl4.fit(x_train_arr, y_train, method,learningRate , 100)
    y_predict = sl4.predict(x_test_arr)
    accuracy = sl4.test(y_test, y_predict)
    print(accuracy)
    print('accuracy:', metrics.accuracy_score(y_test, y_predict))
    print("precision score", precision_score(y_test, y_predict, average='weighted'))
    print("recall score", recall_score(y_test, y_predict, average='weighted'))
    print("F1 score", f1_score(y_test, y_predict, average='weighted'))
    plot_matrix(y_test, y_predict, [0, 1, 2, 3, 4], title='confusion_matrix_svc',
                axis_labels=['negative', 'somewhat negative', 'neutral', 'somewhat positive', 'positive'])

    sl5 = SoftmaxRegression(5, train_features)
    sl5.fit(x_train_arr, y_train, method,learningRate , 1000)
    y_predict = sl5.predict(x_test_arr)
    accuracy = sl5.test(y_test, y_predict)
    print(accuracy)
    print('accuracy:', metrics.accuracy_score(y_test, y_predict))
    print("precision score", precision_score(y_test, y_predict, average='weighted'))
    print("recall score", recall_score(y_test, y_predict, average='weighted'))
    print("F1 score", f1_score(y_test, y_predict, average='weighted'))
    plot_matrix(y_test, y_predict, [0, 1, 2, 3, 4], title='confusion_matrix_svc',
                axis_labels=['negative', 'somewhat negative', 'neutral', 'somewhat positive', 'positive'])

