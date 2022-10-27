from evaluate_accuracy import EvaluateAccuracy
import argparse
from gluonts.env import env
import os
import warnings

warnings.filterwarnings('ignore')
env._push(use_tqdm=False)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CLI = argparse.ArgumentParser()
CLI.add_argument("--microservices", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
CLI.add_argument("--metrics", nargs="*", type=str, default=['cpu', 'memory', 'response_time', 'traffic'])
CLI.add_argument("--learning_algorithms", nargs="*", type=str,
                 default=['arima', 'da-rnn', 'deep-ar', 'deep-state', 'lstm', 'mlp', 'rf', 'svr', 'tft', 'xgboost'])
args = CLI.parse_args()

if not args.microservices or not args.metrics or not args.learning_algorithms:
    print('You need to specify the microservice, metric and learning algorithm')
    print('For example: python3 calculate_costs.py --metric cpu --microservices 1 --learning_algorithm arima')
    print('For example: python3 calculate_costs.py --metric cpu memory --microservices 1 2 --learning_algorithm svr')
    exit()

for microservice in args.microservices:
    for metric in args.metrics:
        time_series_name = 'microservice ' + str(microservice)
        training_level = 'costs'

        ea = EvaluateAccuracy('mse', None, args.learning_algorithms, metric, time_series_name,
                              training_level)
        ea.generate_costs_accuracy()

