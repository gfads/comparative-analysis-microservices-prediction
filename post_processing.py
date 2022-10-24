from pre_processing import Preprocessor


class PickleStructure:
    def __init__(self, horizontal_step, metric_name, model, model_name, preprocessor, training_level, series_name):
        self.folder = series_name + '/' + metric_name + '/' + training_level + '/'
        self.horizontal_step = horizontal_step
        self.model = model
        self.model_key = model_name + str(preprocessor.sliding_window_size)
        self.model_name = model_name
        self.model_path = self.folder + self.model_key
        self.preprocessor = preprocessor
        self.training_level = training_level
        self.series_name = series_name
        self.metric_name = metric_name

        if self.model_name in ['deep-ar', 'deep-state', 'tft']:
            from gluonts.dataset.pandas import PandasDataset
            from pandas import DataFrame
            from gluonts.dataset.util import to_pandas

            ds_test = PandasDataset(DataFrame(self.preprocessor.time_series, columns=['target'])
                                    [len(self.preprocessor.training_set) + len(self.preprocessor.validation_set):])

            true_values = to_pandas(list(ds_test)[0])

            data = []

            for i in range(0, len(true_values) - int(self.preprocessor.sliding_window_size)):
                data.append(PandasDataset([true_values[0:i + int(self.preprocessor.sliding_window_size)]]))

            self.preprocessor.testing_set = data

        save_pickle(self.__dict__)


def save_pickle(df: dict):
    from pickle import dump
    from os import makedirs
    from os.path import dirname

    makedirs(dirname('pickle/' + df['folder']), exist_ok=True)
    dump(df, open('pickle/' + df['model_path'] + '.pkl', 'wb'))


def load_pickle(file_path: str):
    from pickle import load

    return load(open('pickle/' + file_path + '.pkl', 'rb'))


def save_model(metric_name: str, model: object, model_name: str, preprocessor: Preprocessor, training_level: str,
               series_name: str, horizontal_step: int = 1):
    PickleStructure(horizontal_step, metric_name, model, model_name, preprocessor, training_level, series_name)


def measure_models_accuracy(actual: list, predicted: list, accuracy_measure: str = 'mse', **kwargs):
    from sklearn.metrics import mean_squared_error

    weights = kwargs.get('weights')

    if accuracy_measure == 'mse':
        return mean_squared_error(actual, predicted, sample_weight=weights)


def predict_data(data, model, ml_model: str, lags, **kwargs):
    if ml_model[0: 3] == 'svr' or ml_model[0: 3] == 'mlp' or ml_model[0:2] == 'rf':
        return model.predict(data[:, lags])
    elif ml_model[0: 7] == 'xgboost':
        return model.predict(data[:, lags])
    elif ml_model[0: 4] == 'lstm':
        data = data[:, lags]
        data = data.reshape((data.shape[0], data.shape[1], 1))
        return model.predict(data).ravel()
    elif ml_model[0: 3] in ['dee', 'tft']:
        pred = []
        for d in data:
            predictions = model.predict(d)

            for x in predictions:
                pred.append(x.mean[0])

        return pred

    elif ml_model[0: 6] == 'da-rnn':
        import torch

        with torch.no_grad():
            a = model(data.cuda())

        return list(a.cpu().repeat(1, 2).numpy()[:, -1])

    elif ml_model[0: 5] == 'arima':
        arima_forecast = kwargs.get('arima_forecast')
        if arima_forecast:
            if arima_forecast == 'in_sample':
                return model.predict_in_sample(data)[data]
            elif arima_forecast == 'out_sample':
                return model.predict(data + 1)[-1]
        else:
            return model.predict(len(data))
