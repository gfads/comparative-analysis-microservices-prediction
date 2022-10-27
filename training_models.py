from pandas import read_csv
from algorithms_training import training_models
import argparse

CLI = argparse.ArgumentParser()
CLI.add_argument("--microservices", nargs="*", type=int)
CLI.add_argument("--metrics", nargs="*", type=str)
CLI.add_argument("--learning_algorithms", nargs="*", type=str,
                 default=['arima', 'da-rnn', 'deep-ar', 'deep-state', 'lstm', 'mlp', 'rf', 'svr', 'tft', 'xgboost'])
CLI.add_argument("--sliding_window_sizes", nargs="*", type=int, default=[10, 20, 30, 40, 50, 60])
CLI.add_argument("--max_window_size", nargs="*", type=str, default=60)
CLI.add_argument("--training_percentage", nargs="*", type=str, default=0.6)
CLI.add_argument("--validation_percentage", nargs="*", type=str, default=0.2)
args = CLI.parse_args()

if not args.microservices or not args.metrics or not args.learning_algorithms:
    print('You need to specify the microservice and metric')
    print('For example: python3 training_models.py --metric cpu --microservices 1 --learning_algorithm arima')
    exit()

for microservice in args.microservices:
    for metric in args.metrics:
        time_series_name = 'microservice ' + str(microservice)
        time_series = read_csv('time_series/alibaba/' + time_series_name + '/' + metric + '.csv')['value'].values
        training_level = 'hyper_parameter'
        training_percentage = 0.6
        training_models(args.learning_algorithms, args.sliding_window_sizes, time_series, training_level,
                        args.training_percentage, time_series_name, metric, max_sliding_window_size=args.max_window_size,
                        validation_percentage=args.validation_percentage)
