from post_processing import load_pickle
from post_processing import measure_models_accuracy


class EvaluateAccuracy:
    competence_measure: str = ''
    models: list = []
    lags: list = []
    metric: str = ''
    series: str = ''
    training_level: str = ''

    def __init__(self, competence_measure, lags, models, metric, series, training_level):
        self.competence_measure = competence_measure
        self.lags = lags
        self.models = models
        self.metric = metric
        self.series = series
        self.training_level = training_level

    def generate_performance_accuracy(self):
        from pandas import DataFrame
        from os import makedirs
        from os.path import exists

        result = {}

        for model in self.models:
            result[model] = {}
            for lag in self.lags:
                pckl = load_pickle(self.series + '/' + self.metric + '/' + self.training_level + '/' + model + str(lag))

                if pckl['preprocessor'].testing_y is not None:
                    pckl['preprocessor'].testing_set = pckl['preprocessor'].testing_y.reshape(-1, 1)

                result[model][str(lag)] = measure_models_accuracy(pckl['preprocessor'].testing_set[:, -1],
                                                                  pckl['pred_testing'], self.competence_measure)

        df = DataFrame.from_dict(result, orient='index')

        if not exists('results/' + pckl['folder']):
            makedirs('results/' + pckl['folder'])

        df.to_csv('results/' + pckl['folder'] + self.competence_measure + '.csv')
        DataFrame(df.idxmin(axis=1)).transpose().to_csv('results/' + pckl['folder'] + 'lags.csv', index=False)

        s = df.min(axis=1)
        label = df.idxmin(axis=1)
        print('Better ' + self.series + ' '+self.metric+' model: ' + s.idxmin(), label[s.idxmin()])

    def generate_costs_accuracy(self):
        from pandas import DataFrame
        from os import makedirs
        from os.path import exists

        result = {}
        for model in self.models:
            result[model] = {}
            pckl = load_pickle(self.series + '/' + self.metric + '/' + self.training_level + '/' + model)
            result[model]['costs'] = pckl['time_to_fit'] + pckl['time_to_predict']

        if not exists('results/' + pckl['folder']):
            makedirs('results/' + pckl['folder'])

        df = DataFrame.from_dict(result, orient='index').transpose()
        df.to_csv('results/' + pckl['folder'] + 'costs.csv')

        print('Costs '+self.metric+' '+self.series+':')
        print(df.transpose())
