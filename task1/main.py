# coding:utf-8

"""
@Author    :   ShaoCHi
@Date      :   2022/6/3 20:55
@Name      :   feature_engineering.py
@Software  :   PyCharm
"""
import logging
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import data_preprocessing
from feature_engineering import *
import nni

_logger = logging.getLogger('Sentiment-Analysis-On-Movie-Reviews')
_logger.setLevel(logging.INFO)

# load data
data = pd.read_csv('./Data/train.tsv/train.tsv', sep='\t', header=0, index_col='PhraseId')
data = data.loc[:int(len(data) / 4), :]


# 有一些组合无法跑出结果
def main(param):
    global data
    # data preprocessing
    data_duplicate = data_preprocessing.data_preprocess(data, param['method'])
    print(len(generate_feature_vectors(data_duplicate['Phrase'], param['method'])), len(data_duplicate['C']))
    # data split
    x_train, x_test, y_train, y_test = train_test_split(
        generate_feature_vectors(data_duplicate['Phrase'], param['method']),
        data_duplicate['C'],
        test_size=0.3, random_state=1)
    log_model = LogisticRegression(multi_class=param['multi_class'], solver=param['solver'],
                                   max_iter=param['max_iter'])
    param_grid = {
        'tol': [1e-4, 1e-3, 1e-2],
        'C': [0.4, 0.6, 0.8]
    }
    # 4-fold
    grid_search = GridSearchCV(log_model, param_grid, cv=4)
    grid_search.fit(x_train, y_train)

    # 所有标签
    labels = ['0', '1', '2', '3', '4']
    # 利用sklearn中的log_loss()函数计算交叉熵
    sk_log_loss = log_loss(y_test, grid_search.predict(y_train), labels=labels)
    score = grid_search.score(x_test, y_test)

    # accuracy = accuracy_score(y_test, pred_test)  # 准确率
    nni.report_final_result(score)
    _logger.info('Final accuracy reported: %s', score)


if __name__ == '__main__':
    params = {
        'method': 'bag-of-words',
        'multi_class': "multinomial",
        'solver': "newton-cg",
        'max_iter': 200
    }
    tuned_params = nni.get_next_parameter()
    params.update(tuned_params)
    _logger.info('Hyper-parameters: %s', params)
    main(params)
