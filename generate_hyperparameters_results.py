from pandas import read_csv
from algorithms_training import training_models
import argparse

CLI = argparse.ArgumentParser()
CLI.add_argument("--microservices", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
CLI.add_argument("--metrics", nargs="*", type=str, default=['cpu', 'memory', 'response_time', 'traffic'])
CLI.add_argument("--learning_algorithms", nargs="*", type=str,
                 default=['arima', 'da-rnn', 'deep-ar', 'deep-state', 'lstm', 'mlp', 'rf', 'svr', 'tft', 'xgboost'])
CLI.add_argument("--sliding_window_sizes", nargs="*", type=int, default=[10, 20, 30, 40, 50, 60])
args = CLI.parse_args()

if not args.microservices or not args.metrics or not args.learning_algorithms:
    print('You need to specify the microservice, metric and learning algorithm')
    print('For example: python3 calculate_costs.py --metric cpu --microservices 1 --learning_algorithm arima')
    print('For example: python3 calculate_costs.py --metric cpu memory --microservices 1 2 --learning_algorithm svr')
    exit()

for microservice in args.microservices:
    for metric in args.metrics:
        for learning_algorithm in args.learning_algorithms:
            time_series_name = 'microservice ' + str(microservice)
            time_series = read_csv('time_series/alibaba/' + time_series_name + '/' + metric + '.csv')['value'].values
            training_level = 'hyper_parameter'
            training_percentage = 0.6

            ea = EvaluateAccuracy('mse', args.sliding_window_sizes, learning_algorithms, metric, time_series_name,
                                  training_level)

            ea.evaluate_accuracy_monolithic(time_to_fit)

