from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    model_name: str = None
    time_series = ndarray
    normalised_time_series = ndarray
    scaler = MinMaxScaler
    lags = ndarray
    sliding_window_size = int
    max_sliding_window_size = int
    fixed_testing_subsample_size = int
    fixed_validation_subsample_size: int = None
    windowed_time_series = ndarray
    training_set = ndarray
    training_x = ndarray
    training_y = ndarray
    training_percentage = float
    validation_set: ndarray = None
    validation_x: ndarray = None
    validation_y: ndarray = None
    validation_percentage: float = None
    testing_set: ndarray = None
    testing_x: ndarray = None
    testing_y: ndarray = None

    def __init__(self, model_name, time_series, sliding_window_size, training_percentage: float,
                 max_sliding_window_size: int = None, validation_percentage: float = None):
        self.model_name = model_name
        self.time_series = time_series
        self.sliding_window_size = sliding_window_size
        self.training_percentage = training_percentage
        self.max_sliding_window_size = max_sliding_window_size
        self.validation_percentage = validation_percentage
        self.reshape_array_1d_to_2d()
        self.min_max_normalisation()
        self.select_lag_acf()
        self.windowing()

        if self.max_sliding_window_size is None:
            self.split_sample()
        else:
            self.split_sample_with_multiple_windows()

        self.update_model()

    def update_model(self):
        from numpy import array

        if self.model_name == 'da-rnn':
            self.training_x, self.validation_x, self.testing_x = self.ajuste_dataset()
            self.training_y = self.to_tensor(array(self.training_set[:, -1])).unsqueeze(1)
            self.validation_y = self.validation_set[:, -1]
            self.testing_y = self.testing_set[:, -1]

        elif self.model_name in ['deep-ar', 'deep-state', 'tft']:
            from gluonts.dataset.pandas import PandasDataset
            from pandas import DataFrame
            from gluonts.dataset.util import to_pandas

            self.training_x = PandasDataset(
                DataFrame(self.normalised_time_series[:(len(self.training_set) + self.sliding_window_size)],
                          columns=['target']))

            self.training_y = self.training_set[: -1]

            if self.validation_set is not None:
                validation_set = self.normalised_time_series[(len(self.training_set) - self.sliding_window_size):
                                                             len(self.training_set) + len(self.validation_set)]

                true_values = to_pandas(list(PandasDataset(DataFrame(validation_set, columns=['target'])))[0])
                data = []

                for i in range(0, len(true_values) - self.sliding_window_size):
                    data.append(PandasDataset([true_values[i:i + self.sliding_window_size]]))

                self.validation_x = data
                self.validation_y = self.validation_set[:, -1]

                testing_set = self.normalised_time_series[len(self.training_set) + len(self.validation_set):]
                true_values = to_pandas(list(PandasDataset(DataFrame(testing_set, columns=['target'])))[0])
                data = []

                for i in range(0, len(true_values) - self.sliding_window_size):
                    data.append(PandasDataset([true_values[i:i + self.sliding_window_size]]))

                self.testing_x = data
                self.testing_y = self.testing_set[:, -1]

    def reshape_array_1d_to_2d(self):
        """
        This method reshapes a numpy array from 1 to 2 dimensions.
        """
        self.time_series = self.time_series.reshape(-1, 1)

    def min_max_normalisation(self, minimum: float = 0, maximum: float = 1):
        """
        This method normalizes a self.time_series to an interval [a, b] using Sklearn's MinMaxScaler.
        :param minimum: minimum normalisation value
        :param maximum: maximum normalisation value
        """
        from sklearn.preprocessing import MinMaxScaler

        self.scaler = MinMaxScaler(feature_range=(minimum, maximum)).fit(self.time_series)
        self.normalised_time_series = self.scaler.transform(self.time_series)

    def select_lag_acf(self):
        """
        This method finds the best delays of a self.normalised_time_series using ACF considering a
        self.sliding_window_size
        """
        from statsmodels.tsa.stattools import acf
        auto_correlation, confidence_intervals = acf(self.normalised_time_series, nlags=self.sliding_window_size,
                                                     alpha=.05, fft=False)
        upper_threshold = confidence_intervals[:, 1] - auto_correlation
        lower_threshold = confidence_intervals[:, 0] - auto_correlation
        lags = []

        for i in range(1, self.sliding_window_size + 1):
            if auto_correlation[i] >= upper_threshold[i] or auto_correlation[i] <= lower_threshold[i]:
                lags.append(i - 1)

        if len(lags) == 0:
            lags = [i for i in range(self.sliding_window_size)]

        lags = [self.sliding_window_size - (i + 1) for i in lags]
        lags = sorted(lags, key=int)

        self.lags = lags

    def select_lag_pacf(self):
        from statsmodels.tsa.stattools import pacf
        auto_correlation, confidence_intervals = pacf(self.normalised_time_series, nlags=self.sliding_window_size,
                                                      alpha=.05)
        upper_threshold = confidence_intervals[:, 1] - auto_correlation
        lower_threshold = confidence_intervals[:, 0] - auto_correlation
        lags = []

        for i in range(1, self.sliding_window_size + 1):
            if auto_correlation[i] >= upper_threshold[i] or auto_correlation[i] <= lower_threshold[i]:
                lags.append(i - 1)  # -1 por conta que o lag 1 em python é o 0

        if len(lags) == 0:
            lags = [i for i in range(self.sliding_window_size)]

        lags = [self.sliding_window_size - (i + 1) for i in lags]
        lags = sorted(lags, key=int)

        self.lags = lags

    def windowing(self, step_ahead: int = 1, drop_nan: bool = True):
        """
        This method transforms the sample self.normalised_time_series into a set of sliding windows
        (self.windowed_time_series) of size self.sliding_window_size with N steps ahead.
        :param step_ahead: Number of steps ahead
        :param drop_nan: A boolean specified whether null values are to be removed.
        """
        from pandas import DataFrame, concat
        df_ts = DataFrame(self.normalised_time_series)

        cols, names = list(), list()
        for i in range(self.sliding_window_size, 0, -1):
            cols.append(df_ts.shift(i))

        for i in range(0, step_ahead):
            cols.append(df_ts.shift(-i))

        agg = concat(cols, axis=1)

        if drop_nan:
            agg.dropna(inplace=True)

        self.windowed_time_series = agg.values

    def determines_fixed_size_sub_samples(self):
        """
        This method determines a fixed size for the validation and test samples. The training sample size is variable.
        """
        sample_size = len(self.windowed_time_series) - (self.max_sliding_window_size - self.sliding_window_size)

        if self.validation_percentage is None:
            self.fixed_testing_subsample_size = int(sample_size * (1 - self.training_percentage))
        else:
            self.fixed_validation_subsample_size = int(sample_size * self.validation_percentage)
            self.fixed_testing_subsample_size = int(
                sample_size * (1 - (self.training_percentage + self.validation_percentage)))

    def split_sample_with_multiple_windows(self):
        """
        This method splits the self.windowed_time_series sample into self.training_set, self.validation_set and
        self.testing_set with fixed sizes due to the use of different sliding window sizes.
        """
        self.determines_fixed_size_sub_samples()

        training_size = len(self.windowed_time_series) - self.fixed_testing_subsample_size

        if self.fixed_validation_subsample_size is None:
            self.training_set = self.windowed_time_series[:training_size]
            self.testing_set = self.windowed_time_series[training_size:]
        else:
            training_size = training_size - self.fixed_validation_subsample_size
            self.training_set = self.windowed_time_series[:training_size]
            self.validation_set = self.windowed_time_series[
                                  training_size:training_size + self.fixed_validation_subsample_size]
            self.testing_set = self.windowed_time_series[training_size + self.fixed_validation_subsample_size:]

    def split_sample(self):
        """
        This method divides the self.windowed_time_series sample into self.training_set, self.validation_set,
        and self.testing_set.
        """
        training_set_size = round(len(self.windowed_time_series) * self.training_percentage)

        if self.validation_percentage is not None:
            validation_set_size: int = round(len(self.windowed_time_series) * self.validation_percentage)
            superior_validation_set_size = training_set_size + validation_set_size
            self.training_set = self.windowed_time_series[0:training_set_size]
            self.validation_set = self.windowed_time_series[training_set_size:superior_validation_set_size]
            self.testing_set = self.windowed_time_series[superior_validation_set_size:]

        else:
            self.training_set = self.windowed_time_series[0:training_set_size]
            self.testing_set = self.windowed_time_series[training_set_size:]

    def ajuste_dataset(self):
        from numpy import concatenate, array

        if self.validation_set is not None:
            c = concatenate([self.training_set, self.validation_set, self.testing_set])
            c_adjust = array(self.adapt_train_to_(c))
            train = c_adjust[:len(self.training_set[:, 0:-1])]
            valid = c_adjust[len(self.training_set[:, 0:-1]): len(self.training_set[:, 0:-1]) + len(
                self.validation_set[:, 0:-1])]
            test = c_adjust[len(self.training_set[:, 0:-1]) + len(self.validation_set[:, 0:-1]):]

            return self.to_tensor(train), self.to_tensor(valid), self.to_tensor(test)
        else:
            dataset = concatenate([self.training_set, self.testing_set])
            c_adjust = array(self.adapt_train_to_(dataset))
            train = c_adjust[:len(self.training_set[:, 0:-1]), ]
            test = c_adjust[len(self.training_set[:, 0:-1]):, ]

            return self.to_tensor(train), None, self.to_tensor(test)

    def adapt_train_to_(self, dataset):
        from copy import deepcopy

        b = []
        for i in range(0, len(dataset)):
            g = []
            c = deepcopy(i)
            d = 0
            for w in dataset[i]:
                if d in self.lags:
                    a = (c - 0) / ((len(dataset) + len(self.lags)) - 0)

                    g.append([a, w])

                c += 1
                d += 1

            b.append(g)

        return b

    def to_tensor(self, array):
        import torch

        return torch.from_numpy(array).float()
