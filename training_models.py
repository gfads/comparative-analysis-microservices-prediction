from pandas import read_csv
from algorithms_training import training_models
from sys import argv

for m_index in [int(argv[1])]:
    for metric in [argv[2]]:
        time_series_name = 'microservice ' + str(m_index)
        time_series = read_csv('time_series/alibaba/' + time_series_name + '/' + metric + '.csv')['value'].values
        training_level = 'hyper_parameter'
        training_percentage = 0.6
        #learning_algorithms = ['arima', 'da-rnn', 'deep-ar', 'deep-state', 'lstm', 'mlp', 'rf', 'svr', 'tft', 'xgboost']
        learning_algorithms = ['svr']#, 'da-rnn', 'deep-ar', 'deep-state', 'lstm', 'mlp', 'rf', 'svr', 'tft', 'xgboost']
        #sliding_window_sizes = [10, 20, 30, 40, 50, 60]
        sliding_window_sizes = [10, 20, 30, 40, 50, 60]
        training_models(learning_algorithms, sliding_window_sizes, time_series, training_level, training_percentage,
                        time_series_name, metric, validation_percentage=0.2)
