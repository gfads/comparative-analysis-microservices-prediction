from itertools import product


def training_models(learning_algorithms, sliding_window_sizes, time_series, training_level, training_percentage,
                    serie_key, metric_name, learning_parameters=None, validation_percentage=None,
                    max_sliding_window_size=60):
    from post_processing import save_model
    from preprocessor import Preprocessor
    import time

    for mn in learning_algorithms:
        for ws in sliding_window_sizes:
            start_time = time.time()

            pp = Preprocessor(mn, time_series, ws, training_percentage, validation_percentage=validation_percentage,
                              max_sliding_window_size=max_sliding_window_size)
            model = select_model(mn, training_level, pp, learning_parameters=learning_parameters)

            if learning_parameters is not None:
                time_to_fit = time.time() - start_time
                save_model(metric_name, model, mn, pp, training_level, serie_key, time_to_fit=time_to_fit)
            else:
                save_model(metric_name, model, mn, pp, training_level, serie_key)


def d_values(data: list):
    a = 0
    for index in range(len(data) - 1, 0, -1):
        if (data[index] - data[index - 1]) != 1:
            return len(data) - 1 - index
        else:
            a = len(data) - 1

    return a


def find_p_d_q_arima(pp):
    from pmdarima.arima import ADFTest

    adf_test = ADFTest(alpha=0.05)
    dtr = adf_test.should_diff(pp.normalised_time_series)
    d = 0

    if dtr[1]:
        d = 1

    pp.select_lag_acf()
    q = d_values(pp.lags)

    pp.select_lag_pacf()
    p = d_values(pp.lags)

    return p, d, q


class ARIMA:
    models: list = None
    model_type: str = 'arima'

    def __init__(self, training_level):
        self.training_level = training_level

        if self.training_level in ['hyper_parameter', 'costs']:
            self.standard_training()

    def standard_training(self):
        from pmdarima import arima
        self.models = arima


class DARNN:
    models: list = []
    model_type: str = 'da-rnn'
    enconder = [16, 32, 64, 128, 256]
    sliding_window_size: int = None

    def __init__(self, training_level, sliding_window_size, lp):
        self.training_level = training_level
        self.sliding_window_size = sliding_window_size
        self.lp = lp

        if self.training_level == 'hyper_parameter':
            self.hyperparameter_training()
        elif self.training_level == 'costs':
            self.costs_training()

    def costs_training(self):
        from da_rnn.torch import DARNN

        self.models = DARNN(n=1, T=len(self.sliding_window_size), m=self.lp[0], p=self.lp[0], y_dim=1, dropout=0)

    def hyperparameter_training(self):
        from da_rnn.torch import DARNN

        self.models = []

        for e in self.enconder:
            self.models.append(DARNN(n=1, T=len(self.sliding_window_size), m=e, p=e, y_dim=1, dropout=0))


class DeepAR:
    models: list = []
    model_type: str = 'deep-ar'
    encoder = [8]
    decoder = [8]
    batch = [64]
    learning_rate = [0.0001]
    layers = [3]
    lstm_nodes = [40]

    def __init__(self, training_level):
        self.training_level = training_level

        if self.training_level == 'costs':
            self.costs_training()
        elif self.training_level == 'hyper_parameter':
            self.hyperparameter_training()

    def costs_training(self):
        from gluonts.mx import Trainer
        from gluonts.model.deepar import DeepAREstimator

        self.models = DeepAREstimator(prediction_length=1, freq="M", trainer=Trainer(epochs=1), batch_size=256)

    def hyperparameter_training(self):
        from gluonts.mx import Trainer
        from gluonts.model.deepar import DeepAREstimator

        self.models = [DeepAREstimator(num_layers=3, num_cells=40, prediction_length=1, freq="M",
                                       trainer=Trainer(), batch_size=64)]


class DeepState:
    models: list = []
    model_type: str = 'deep-state'

    def __init__(self, training_level):
        self.training_level = training_level

        if self.training_level == 'costs':
            self.standard_training()
        elif self.training_level == 'hyper_parameter':
            self.hyperparameter_training()

    def standard_training(self):
        from gluonts.mx import Trainer
        from gluonts.model.deepstate import DeepStateEstimator

        self.models = DeepStateEstimator(prediction_length=1, freq="M", trainer=Trainer(), cardinality=[1],
                                         use_feat_static_cat=False)

    def hyperparameter_training(self):
        from gluonts.mx import Trainer
        from gluonts.model.deepstate import DeepStateEstimator

        self.models = [DeepStateEstimator(prediction_length=1, freq="M", trainer=Trainer(), cardinality=[1],
                                          use_feat_static_cat=False)]


class LSTM:
    models: list = []
    model_type: str = 'lstm'
    batch_size = [64, 128]
    epochs = [1, 2, 4, 8, 10]
    hidden_layers = [2, 3, 4, 5, 6]
    learning_rate = [0.05, 0.01, 0.001]
    number_of_units = [50, 75, 125]

    hyper_param = list(product(batch_size, epochs, hidden_layers, learning_rate, number_of_units))

    def __init__(self, training_level):
        self.training_level = training_level

        if self.training_level == 'costs':
            self.costs_training()
        elif self.training_level == 'hyper_parameter':
            self.hyperparameter_training()

    def costs_training(self):
        from keras.models import Sequential

        self.models = Sequential()

    def hyperparameter_training(self):
        from keras.models import Sequential
        import random

        self.models = []
        self.hyper_param = random.sample(self.hyper_param, 60)

        for bs, e, hl, lr, nu in self.hyper_param:
            self.models.append(
                [Sequential(), {'batch_size': bs, 'epochs': e, 'hidden_layers': hl, 'learning_rate': lr,
                                'number_of_units': nu}])


class MLP:
    models: list = []
    model_type: str = 'sklearn'
    hidden_layer_sizes = [2, 5, 10, 15, 20]
    activation = ['logistic']
    solver = ['adam']
    num_exec = 10

    hyper_param = list(product(hidden_layer_sizes, activation, solver, range(0, num_exec)))

    def __init__(self, training_level, lp):
        self.training_level = training_level
        self.lp = lp

        if self.training_level == 'hyper_parameter':
            self.hyperparameter_training()

        elif self.training_level == 'costs':
            self.costs_training()

    def costs_training(self):
        from sklearn.neural_network import MLPRegressor
        self.models = MLPRegressor(activation=self.lp[0], hidden_layer_sizes=self.lp[1], max_iter=1000)

    def hyperparameter_training(self):
        from sklearn.neural_network import MLPRegressor

        self.models = []

        for hls, a, s, mi, _ in self.hyper_param:
            self.models.append(
                MLPRegressor(hidden_layer_sizes=hls, activation=a, solver=s, max_iter=1000))


class RF:
    models: list = []
    model_type: str = 'sklearn'
    min_samples_leaf = [1, 5, 10]
    min_samples_split = [2, 5, 10, 15]
    n_estimators = [100, 500, 1000]
    hyper_param = list(product(min_samples_leaf, min_samples_split, n_estimators))

    def __init__(self, training_level, lp):
        self.training_level = training_level
        self.lp = lp

        if self.training_level == 'hyper_parameter':
            self.hyperparameter_training()
        elif self.training_level == 'costs':
            self.costs_training()

    def hyperparameter_training(self):
        from sklearn.ensemble import RandomForestRegressor

        self.models = []

        for msl, mss, ne in self.hyper_param:
            self.models.append(RandomForestRegressor(n_estimators=ne, min_samples_leaf=msl, min_samples_split=mss,
                                                     n_jobs=-1))

    def costs_training(self):
        from sklearn.ensemble import RandomForestRegressor
        self.models = RandomForestRegressor(min_samples_leaf=self.lp[0], min_samples_split=self.lp[1],
                                            n_estimators=self.lp[2], n_jobs=-1)


class SVR:
    models: list = []
    model_type: str = 'sklearn'
    training_level: str = None

    gamma: list = [0.001, 0.01, 0.1, 1]
    kernel: list = ['rbf', 'sigmoid']
    epsilon: list = [0.1, 0.001, 0.0001]
    C: list = [0.1, 1, 10, 100, 1000, 10000]

    hyper_param = list(product(kernel, gamma, epsilon, C))

    def __init__(self, training_level, lp):
        self.training_level = training_level
        self.lp = lp

        if self.training_level == 'hyper_parameter':
            self.hyperparameter_training()
        elif self.training_level == 'costs':
            self.costs_training()

    def hyperparameter_training(self):
        from sklearn.svm import SVR
        import random

        self.models = []
        self.hyper_param = random.sample(self.hyper_param, 3)

        for k, g, e, C in self.hyper_param:
            self.models.append(SVR(C=C, epsilon=e, kernel=k, gamma=g))

    def costs_training(self):
        from sklearn.svm import SVR
        self.models = SVR(C=self.lp[0], epsilon=self.lp[1], gamma=self.lp[2], kernel=self.lp[3])


class TFT:
    models: list = []
    model_type: str = 'tft'
    dropout_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    learning_rate = [0.0001, 0.001, 0.01]
    num_heads = [1, 4]
    batch = [64, 128, 256]

    hyper_param = list(product(batch, dropout_rate, learning_rate, num_heads))

    def __init__(self, training_level):
        self.training_level = training_level

        if self.training_level == 'costs':
            self.costs_training()
        elif self.training_level == 'hyper_parameter':
            self.hyperparameter_training()

    def costs_training(self):
        from gluonts.mx import Trainer
        from gluonts.mx.trainer.learning_rate_scheduler import LearningRateReduction
        from gluonts.model.tft import TemporalFusionTransformerEstimator

        callbacks = [LearningRateReduction(objective="min", patience=9, base_lr=1e-3, decay_factor=0.5)]

        self.models = TemporalFusionTransformerEstimator(freq="M", prediction_length=1,
                                                         trainer=Trainer(epochs=1, batch_size=256,
                                                                         num_batches_per_epoch=256,
                                                                         callbacks=callbacks),
                                                         )

    def hyperparameter_training(self):
        from gluonts.mx import Trainer
        from gluonts.model.tft import TemporalFusionTransformerEstimator
        from gluonts.mx.trainer.learning_rate_scheduler import LearningRateReduction
        from random import sample

        self.models = []

        self.hyper_param = sample(self.hyper_param, 60)

        for b, dp, lr, nh in self.hyper_param:
            callbacks = [LearningRateReduction(objective="min", patience=9, base_lr=lr, decay_factor=0.5)]

            self.models.append(
                TemporalFusionTransformerEstimator(freq="M", prediction_length=1, num_heads=nh, dropout_rate=dp,
                                                   trainer=Trainer(batch_size=b, num_batches_per_epoch=b,
                                                                   callbacks=callbacks), ))


class XGBoost:
    models: list = []
    model_type: str = 'xgboost'

    col_sample_by_tree = [0.4, 0.6, 0.8]
    gamma = [1, 5, 10]
    learning_rate = [0.01, 0.1, 1]
    max_depth = [3, 6, 10]
    n_estimators = [100, 150, 200]
    reg_alpha = [0.01, 0.1, 10]
    reg_lambda = [0.01, 0.1, 10]
    subsample = [0.4, 0.6, 0.8]

    hyper_param = list(
        product(col_sample_by_tree, gamma, learning_rate, max_depth, n_estimators, reg_alpha, reg_lambda, subsample))

    def __init__(self, training_level, lp):
        self.training_level = training_level
        self.lp = lp

        if self.training_level == 'hyper_parameter':
            self.hyperparameter_training()
        elif self.training_level == 'costs':
            self.costs_training()

    def hyperparameter_training(self):
        from xgboost import XGBRegressor
        import random

        self.models = []

        self.hyper_param = random.sample(self.hyper_param, 60)

        for csb, g, lr, md, ne, ra, rl, ss in self.hyper_param:
            self.models.append(XGBRegressor(colsample_bytree=csb, gamma=g, learning_rate=lr, max_depth=md,
                                            n_estimators=ne, reg_alpha=ra, reg_lambda=rl, subsample=ss))

    def costs_training(self):
        from xgboost import XGBRegressor
        self.models = XGBRegressor(colsample_bytree=self.lp[0], gamma=self.lp[1], learning_rate=self.lp[2],
                                   max_depth=self.lp[3], n_estimators=self.lp[4], reg_alpha=self.lp[5],
                                   reg_lambda=self.lp[6])


def select_model(model_name, training_level, preprocessor, learning_parameters=None):
    mt = ModelTraining()
    model = None

    if model_name == 'arima':
        model = ARIMA(training_level)
    elif model_name == 'da-rnn':
        model = DARNN(training_level, preprocessor.lags, learning_parameters)
    elif model_name == 'deep-ar':
        model = DeepAR(training_level)
    elif model_name == 'deep-state':
        model = DeepState(training_level)
    elif model_name == 'lstm':
        model = LSTM(training_level)
    elif model_name == 'mlp':
        model = MLP(training_level, learning_parameters)
    elif model_name == 'rf':
        model = RF(training_level, learning_parameters)
    elif model_name == 'svr':
        model = SVR(training_level, learning_parameters)
    elif model_name == 'tft':
        model = TFT(training_level)
    elif model_name == 'xgboost':
        model = XGBoost(training_level, learning_parameters)

    mt.train_models(preprocessor, model, learning_parameters)

    return mt.trained_model


def train_arima(model, normalised_time_series, training_set, validation_set, window_size):
    data = normalised_time_series[:(len(training_set) + len(validation_set) + window_size)]

    return model.auto_arima(data)


def train_arima_costs(model, normalised_time_series, training_set, validation_set, window_size, lp):
    data = normalised_time_series[:(len(training_set) + len(validation_set) + window_size)]

    return model.auto_arima(data, start_p=lp[0], max_p=lp[0], start_q=lp[2], max_q=lp[2])


def train_da_rnn(model, training_x, training_y):
    from da_rnn.torch import DEVICE
    from poutyne import Model, EarlyStopping

    for _ in range(0, 30):
        d = Model(model, 'adam', 'mse', device=DEVICE)
        callbacks = [EarlyStopping(monitor='loss', patience=5)]
        d.fit(training_x, training_y, epochs=10, batch_size=256, callbacks=callbacks, verbose=False)

    return model


def train_lstm(lags, model, training_set, batch_size=256, epochs=10, learning_rate=0.001,
               number_of_units=100, hidden_layers=1):
    from keras.layers import LSTM, Dense
    import tensorflow as tf

    x_training = training_set[:, lags]
    y_training = training_set[:, -1]
    x_training = x_training.reshape((x_training.shape[0], x_training.shape[1], 1))

    for _ in range(0, hidden_layers):
        model.add(LSTM(number_of_units, activation='relu', return_sequences=True, input_shape=(len(lags), 1)))

    model.add(LSTM(number_of_units, activation='relu'))
    model.add(Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')
    model.fit(x_training, y_training, epochs=epochs, batch_size=batch_size, verbose=0)

    return model


def train_sklearn(lags, model, training_set):
    a = None
    for _ in range(0, 30):
        a = model.fit(training_set[:, lags], training_set[:, -1])

    return a


def train_hyper_parameter_da_rnn(models, training_x, training_y):
    aux_model = []

    for model in models:
        aux_model.append(train_da_rnn(model, training_x, training_y))

    return aux_model


def train_hyper_parameter_arima(model, normalised_time_series, training_set, validation_set, window_size, pp):
    pp.normalised_time_series = normalised_time_series[:(len(training_set) + len(validation_set) + window_size)]
    p, d, q = find_p_d_q_arima(pp)

    if d == 0:
        return model.auto_arima(pp.normalised_time_series, start_p=0, start_q=0, max_p=p, max_q=q, seasonal=False,
                                error_action='warn', trace=False, suppress_warnings=True, stepwise=True)
    else:
        return model.auto_arima(pp.normalised_time_series, start_p=0, d=d, start_q=0, max_p=p, max_d=2, max_q=q,
                                seasonal=False, error_action='warn', trace=False, suppress_warnings=True, stepwise=True)


def train_hyper_parameters_lstm(lags, models, training_set):
    aux_model = []

    for m in models:
        aux_model.append(train_lstm(lags, m[0], training_set, epochs=m[1]['epochs'],
                                    batch_size=m[1]['batch_size'], learning_rate=m[1]['learning_rate'],
                                    number_of_units=m[1]['number_of_units'],
                                    hidden_layers=m[1]['hidden_layers']))

    return aux_model


def train_xgboost(lags, model, training_set):
    a = None
    for _ in range(0, 30):
        a = model.fit(training_set[:, lags], training_set[:, -1])

    return a


def train_deep_models(model, training_x):
    a = None
    for _ in range(0, 30):
        a = model.train(training_x)
    return a


def train_hyper_parameters(lags, models, training_set):
    aux_model = []

    for model in models:
        aux_model.append(train_sklearn(lags, model, training_set))

    return aux_model


def train_hyper_parameters_xgboost(lags, models, training_set):
    aux_model = []

    for model in models:
        aux_model.append(train_xgboost(lags, model, training_set))

    return aux_model


def train_hyper_parameters_deep_models(models, training_set):
    aux_model = []

    for model in models:
        aux_model.append(train_deep_models(model, training_set))

    return aux_model


class ModelTraining:
    trained_model = None

    def __init__(self):
        self.trained_models = None

    def train_models(self, pp, model, lp):
        models = model.models
        model_type = model.model_type
        training_level = model.training_level
        lags = pp.lags
        training_set = pp.training_set
        validation_set = pp.validation_set
        sliding_window_size = pp.sliding_window_size
        normalised_time_series = pp.normalised_time_series

        if model_type == 'sklearn':
            if training_level == 'costs':
                self.trained_model = train_sklearn(lags, models, training_set)
            elif training_level == 'hyper_parameter':
                trained_models = train_hyper_parameters(lags, models, training_set)
                self.select_the_most_accurate_model(lags, trained_models, validation_set)
        elif model_type == 'xgboost':
            if training_level == 'costs':
                self.trained_model = train_xgboost(lags, models, training_set)
            elif training_level == 'hyper_parameter':
                trained_models = train_hyper_parameters_xgboost(lags, models, training_set)
                self.select_the_most_accurate_model_xgboost(lags, trained_models, validation_set)
        elif model_type == 'arima':
            if training_level == 'hyper_parameter':
                self.trained_model = train_hyper_parameter_arima(models, normalised_time_series, training_set,
                                                                 validation_set, sliding_window_size, pp)

            elif training_level == 'costs':
                for _ in range(0, 30):
                    self.trained_model = train_arima_costs(models, normalised_time_series, training_set,
                                                           validation_set, sliding_window_size, lp)

        elif model_type == 'lstm':
            if training_level == 'costs':
                from keras.models import Sequential
                for _ in range(0, 30):
                    self.trained_model = train_lstm(lags, Sequential(), training_set, hidden_layers=lp[0],
                                                    number_of_units=lp[1], learning_rate=lp[2])

            elif training_level == 'hyper_parameter':
                trained_models = train_hyper_parameters_lstm(lags, models, training_set)
                self.select_the_most_accurate_model_lstm(lags, trained_models, validation_set)

        elif model_type in ['deep-ar', 'deep-state', 'tft']:
            if training_level == 'costs':
                self.trained_model = train_deep_models(models, pp.training_x)

            elif training_level == 'hyper_parameter':
                trained_models = train_hyper_parameters_deep_models(models, pp.training_x)
                self.select_the_most_accurate_model_deep_models(lags, trained_models, pp.validation_x,
                                                                pp.validation_y)

        elif model_type == 'da-rnn':
            if training_level == 'costs':
                self.trained_model = train_da_rnn(models, pp.training_x, pp.training_y)
            elif training_level == 'hyper_parameter':
                trained_models = train_hyper_parameter_da_rnn(models, pp.training_x, pp.training_y)
                self.select_the_most_accurate_model_da_rnn_models(lags, trained_models, pp.validation_x,
                                                                  pp.validation_y)

        return self.trained_model

    def select_the_most_accurate_model(self, lags, models, validation_set, competence_measure='mse'):
        from post_processing import measure_models_accuracy
        from numpy import Inf

        best_model = None
        best_accuracy = Inf

        for model in models:
            predicted = model.predict(validation_set[:, lags])
            accuracy_metric = measure_models_accuracy(validation_set[:, -1], predicted, competence_measure)

            if accuracy_metric < best_accuracy:
                best_accuracy = accuracy_metric
                best_model = model

        self.trained_model = best_model

    def select_the_most_accurate_model_xgboost(self, lags, models, validation_set, competence_measure='mse'):
        from post_processing import measure_models_accuracy, predict_data
        from numpy import Inf

        best_model = None
        best_accuracy = Inf

        for model in models:
            predicted = predict_data(validation_set, model, 'xgboost', lags)
            accuracy_metric = measure_models_accuracy(validation_set[:, -1], predicted, competence_measure)

            if accuracy_metric < best_accuracy:
                best_accuracy = accuracy_metric
                best_model = model

        self.trained_model = best_model

    def select_the_most_accurate_model_lstm(self, lags, models, validation_set, competence_measure='mse'):
        from post_processing import measure_models_accuracy, predict_data
        from numpy import Inf, isnan

        best_model = None
        best_accuracy = Inf

        for model in models:
            predicted = predict_data(validation_set, model, 'lstm', lags)

            if isnan(predicted).all():
                accuracy_metric = 1000
            else:
                accuracy_metric = measure_models_accuracy(validation_set[:, -1], predicted, competence_measure)

            if accuracy_metric < best_accuracy:
                best_accuracy = accuracy_metric
                best_model = model

        self.trained_model = best_model

    def select_the_most_accurate_model_deep_models(self, lags, models, validation_x, validation_y,
                                                   competence_measure='mse'):
        from post_processing import measure_models_accuracy, predict_data
        from numpy import Inf

        best_model = None
        best_accuracy = Inf

        for model in models:
            predicted = predict_data(validation_x, model, 'tft', lags)
            accuracy_metric = measure_models_accuracy(validation_y, predicted, competence_measure)

            if accuracy_metric < best_accuracy:
                best_accuracy = accuracy_metric
                best_model = model

        self.trained_model = best_model

    def select_the_most_accurate_model_da_rnn_models(self, lags, models, validation_x, validation_y,
                                                     competence_measure='mse'):
        from post_processing import measure_models_accuracy, predict_data
        from numpy import Inf

        best_model = None
        best_accuracy = Inf

        for model in models:
            predicted = predict_data(validation_x, model, 'da-rnn', lags)
            accuracy_metric = measure_models_accuracy(validation_y, predicted, competence_measure)

            if accuracy_metric < best_accuracy:
                best_accuracy = accuracy_metric
                best_model = model

        self.trained_model = best_model
