# Predictive models for adapting microservice-based applications: a comparative analysis
 
## Project Description

This repository contains the implementation and experimental results of a comparative analysis of statistical, machine learning (ML), and deep learning (DL) algorithms for forecasting microservice time series.

The study evaluates ten algorithms: one statistical algorithm, Autoregressive Integrated Moving Average (ARIMA); five deep learning algorithms, Dual-Stage Attention-Based Recurrent Neural Network (DA-RNN), Deep State Space Model (DeepState), DeepAR, Long Short-Term Memory (LSTM), and Temporal Fusion Transformer (TFT); and four traditional machine learning algorithms, Multilayer Perceptron (MLP), Support Vector Regression (SVR), Random Forest (RF), and Extreme Gradient Boosting (XGBoost).


## Dataset Description

The experiments use 40 time series extracted from production microservices in the large-scale Alibaba cluster deployment available in the [Alibaba Cluster Trace dataset](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2021).

The repository provides the selected time series and supporting information in the following directories:

- `time_series/alibaba/`: Contains the 40 time series used in the experiments.
- `other_results/data-descriptions/`: Contains descriptive information about the selected time series, including their characteristics, microservice and container IDs, and seasonality.
- `other_results/dtw/`: Contains the Dynamic Time Warping (DTW) distances between the selected time series.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```
Install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
## Project Files

Summary of the main repository files and directories.

| Repository Path | Content Description |
|---|---|
| [`other_results/data-descriptions/`](other_results/data-descriptions/) | Descriptive information about the selected time series, including their characteristics, microservice and container IDs, and seasonality. |
| [`other_results/dtw/`](other_results/dtw/) | Dynamic Time Warping (DTW) distances between the selected time series. |
| [`other_results/friedman-test/`](other_results/friedman-test/) | Results of the Friedman and Nemenyi tests for each microservice metric, used to compare the forecasting performance of the evaluated algorithms. |
| [`other_results/hyperparameters/`](other_results/hyperparameters/) | Selected hyperparameter configurations for each forecasting model. |
| [`other_results/plots/`](other_results/plots/) | Plots illustrating the forecasting results of the models across different lag configurations. |
| [`results/`](results/) | Mean Squared Error (MSE) and Model Efficiency Time (MET) results for the evaluated models. |
| [`pickle/`](pickle/) | Trained models serialized and stored as pickle files. |

## Generating Experimental Results

The following scripts generate the experimental results using the trained models stored in the `pickle` directory.

|  File                                | File description                                                         |
|--------------------------------------|--------------------------------------------------------------------------|
| generate_met_results.py              | Generates MET results from models.                                       |
| generate_mse_results.py              | Generates MSE results from models.                                       |

## Model Hyperparameters

The table below summarizes the hyperparameters and search spaces considered during model training. The selected hyperparameter configurations for each model are available in `other_results/hyperparameters/`.

| Algorithms with Source                                         |  Hyperparameters                                                             |
|:------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [ARIMA](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html)  |      AutoARIMA           |
| [DARNN](https://dl.acm.org/doi/10.5555/3172077.3172254)     | 'encoder': (16, 32, 64, 128, 256), 'decoder': (16, 32, 64, 128, 256) |
| [DeepAR](https://doi.org/10.1016/j.ijforecast.2019.07.001)  | 'encoder': (8), 'decoder': (8), 'batch': (64), 'learning\_rate': (0.0001), 'layers': (3), 'lstm\_nodes': (40)                              |
| [DeepState](https://dl.acm.org/doi/10.5555/3327757.3327876) | The algorithm itself selects the hyperparameters                                          |
| [LSTM](https://doi.org/10.1109/JIOT.2020.2964405)  | 'batch\_size': (64, 128), 'epochs': (1, 2, 4, 8, 10), 'hidden\_layers': (2, 3, 4, 5, 6), 'learning\_rate': (0.05, 0.01, 0.001)  |
| [MLP](https://doi.org/10.1109/TNNLS.2021.3051384) | 'hidden\_layer\_sizes': (2, 5, 10, 15, 20), 'activation': ('logistic'), 'solver': ('adam'), 'max\_iter': (1000), 'num\_exec': 10                                                                                                                                         |
| [RF](https://doi.org/10.1016/j.asoc.2021.107850) | 'min\_samples\_leaf': (1, 5, 10), 'min\_samples\_split': (2, 5, 10, 15), 'n\_estimators': (100, 500, 1000)                                                                                                                                                                   |
| [TFT](https://doi.org/10.1016/j.ijforecast.2021.03.012)  | 'dropout\_rate': (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9), 'learning\_rate': (0.0001, 0.001, 0.01), 'num\_heads': (1, 4), 'batch': (64, 128, 256)                                                                                                                             |
| [SVR](https://doi.org/10.1109/TNNLS.2021.3051384) | 'gamma': (0.001,  0.01,  0.1, 1) 'kernel': ('rbf', 'sigmoid') 'epsilon': (0.1, 0.001, 0.0001) 'C': (0.1, 1, 10, 100, 1000, 10000)                                                                                                                                        |
| [XGBoost](https://doi.org/10.32734/jocai.v5.i2-6290)  | 'col\_sample\_by\_tree': (0.4, 0.6, 0.8), 'gamma': (1, 5, 10), 'learning\_rate': (0.01, 0.1, 1), 'max\_depth': (3, 6, 10), 'n\_estimators': (100, 150, 200), 'reg\_alpha': (0.01, 0.1, 10), 'reg\_lambda': (0.01, 0.1, 10), 'subsample': (0.4, 0.6, 0.8) |

## Experimental Environment

### Hardware Configuration

The experiments were conducted on a Dell Vostro 5490 laptop equipped with the following hardware:

- **Processor:** Intel Core i7-10510U (4 cores and 8 threads).
- **RAM:** 16 GB.
- **GPU:** NVIDIA GeForce MX230 (2 GB).
- **Integrated Graphics:** Intel UHD Graphics.

The system ran Linux Mint 20.3 (Una), based on Ubuntu 20.04 LTS, with Linux kernel 5.4.0-216-generic.

### Software Configuration

The experimental environment used Python 3.8.10 and the following main software libraries and frameworks:

| Software | Version |
|---|---:|
| TensorFlow | 2.10.0 |
| Keras | 2.10.0 |
| PyTorch | 1.12.1+cu102 |
| scikit-learn | 1.1.2 |
| NumPy | 1.23.1 |
| GluonTS | 0.10.2 |
| XGBoost | 2.1.3 |
| pmdarima | 1.8.5 |

The complete list of dependencies and their respective versions is available in the `requirements.txt` file.

### GPU Acceleration

The LSTM and DA-RNN implementations were executed on the NVIDIA GeForce MX230 GPU using CUDA. All other algorithms were executed on the CPU because GPU acceleration was not supported at the time, either due to limitations of the respective software libraries or hardware compatibility issues.

## Citation

If you use this repository in your research, please cite the associated publication once it becomes available. The bibliographic reference and DOI will be added to this section upon publication.

<!-- ```bibtex
@article{author2026predictive,
  title   = {Predictive Models for Adapting Microservice-Based Applications: A Comparative Analysis},
  author  = {Author names},
  journal = {Journal name},
  year    = {2026},
  doi     = {DOI}
}
``` -->

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.