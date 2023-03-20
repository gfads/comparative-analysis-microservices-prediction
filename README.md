# Comparative Analisys Microservices Prediction
 

 ## Project description
 
 This study compares popular Machine Learning (ML), Deep Learning (DP), and statistical algorithms for forecasting microservice time series. The evaluated algorithms are a statistical algorithm (AutoRegressive Integrated Moving Average (ARIMA)), five DL ones (Dual-Stage Attention-Based RNN (DARNN), Deep State Space Model (DeepState), DeepAR, Long Short-Term Memory (LSTM), and Temporal Fusion Transformer (TFT)), and four traditional ML (Multilayer Perceptron (MLP), Support Vector Regressor (SVR), Random Forest (RF), and eXtreme Gradient Boosting (XGBoost)). They are evaluated in [40-time series](time_series/alibaba) extracted from microservices operating in production in a large-scale deployment within the [Alibaba Cluster](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2021).
 
# Installation  
  
## How to install the project?

    $ virtualenv venv
    $ source venv/bin/activate
    $ pip3 install -r requirements.txt
    
## Project Files

Summary of the main repository files.


| Files                     | Content description                                                              |
|----------------------------|----------------------------------------------------------------------------------|
| [Data-descriptions.csv](blob/main/other_results/data-descriptions/data-descriptions.csv)                | Description of the datasets.         |
| [DTW](other_results/dtw)   | Result of the DTW algorithm for selecting time series.                           |
| [Friedman and Nemenyi tests](other_results/friedman-test)                   | Friedman and Nemenyi results    |
| [Result](results)          | MSE and model efficiency time (MET) of the models.                               |
| [Models](pickle)           | Trained models saved in .pickle                                                  |


## How to regenerate results using pickle models?

|  File                                | File description                                                         |
|--------------------------------------|--------------------------------------------------------------------------|
| generate_met_results.py            | Generates MET results from models.                                         |
| generate_mse_results.py  | Generates MSE results from models.                                                   |

## Parameters models

The parameters adopted into models’ training is summarises below:

| Algorithms with Source                                         |  Hyperparameters                                                             |
|:------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [ARIMA](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html)  |      AutoArima           |
| [DARNN](https://dl.acm.org/doi/10.5555/3172077.3172254)     | 'enconder': (16, 32, 64, 128, 256), 'decoder': (16, 32, 64, 128, 256) |
| [DeepAr](https://doi.org/10.1016/j.ijforecast.2019.07.001)  | 'encoder': (8), 'decoder': (8), 'batch': (64), 'learning\_rate': (0.0001), 'layers': (3), 'lstm\_nodes': (40)                              |
| [DeepState](https://dl.acm.org/doi/10.5555/3327757.3327876) | The algorithm itself selects the hyperparameters                                          |
| [LSTM](https://doi.org/10.1109/JIOT.2020.2964405)  | 'batch\_size': (64, 128), 'epochs': (1, 2, 4, 8, 10), 'hidden\_layers': (2, 3, 4, 5, 6), 'learning\_rate': (0.05, 0.01, 0.001)  |
| [MLP](https://doi.org/10.1109/TNNLS.2021.3051384) | 'hidden\_layer\_sizes': (2, 5, 10, 15, 20), 'activation': ('logistic'), 'solver': ('adam'), 'max\_iter': (1000), 'num\_exec': 10                                                                                                                                         |
| [RF](https://doi.org/10.1016/j.asoc.2021.107850) | 'min\_samples\_leaf': (1, 5, 10), 'min\_samples\_split': (2, 5, 10, 15), 'n\_estimators': (100, 500, 1000)                                                                                                                                                                   |
| [TFT](https://doi.org/10.1016/j.ijforecast.2021.03.012)  | 'dropout\_rate': (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9), 'learning\_rate': (0.0001, 0.001, 0.01), 'num\_heads': (1, 4), 'batch': (64, 128, 256)                                                                                                                             |
| [SVR](https://doi.org/10.1109/TNNLS.2021.3051384) | 'gamma': (0.001,  0.01,  0.1, 1) 'kernel': ('rbf', 'sigmoid') 'epsilon': (0.1, 0.001, 0.0001) 'C': (0.1, 1, 10, 100, 1000, 10000)                                                                                                                                        |
| [XGBoost](https://doi.org/10.32734/jocai.v5.i2-6290)  | 'col\_sample\_by\_tree': (0.4, 0.6, 0.8), 'gamma': (1, 5, 10), 'learning\_rate': (0.01, 0.1, 1), 'max\_depth': (3, 6, 10), 'n\_estimators': (100, 150, 200), 'reg\_alpha': (0.01, 0.1, 10), 'reg\_lambda': (0.01, 0.1, 10), 'subsample': (0.4, 0.6, 0.8) |
