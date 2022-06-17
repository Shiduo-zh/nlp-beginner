# coding:utf-8

"""
@Author    :   ShaoCHi
@Date      :   2022/6/4 11:53
@Name      :   launch.py
@Software  :   PyCharm
"""

from pathlib import Path

from nni.experiment import Experiment

search_space = {
    "method": {"_type": "choice", "_value": ["bag-of-words", "N-gram"]},
    "multi_class": {"_type": "choice", "_value": ["ovr", "multinomial"]},
    "solver": {"_type": "choice", "_value": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]},
    "max_iter": {"_type": "choice", "_value": [50, 100, 200]}
}

experiment = Experiment('local')
experiment.config.experiment_name = 'Sentiment-Analysis-On-Movie-Reviews'
experiment.config.trial_concurrency = 5
experiment.config.max_trial_number = 10
experiment.config.search_space = search_space
experiment.config.trial_command = 'python main.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.training_service.use_active_gpu = True

experiment.run(8888)
