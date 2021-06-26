# XPBoost
This is the implementation of the paper 'eXtreme Performance Boosting: Revisiting Statistics and Intelligent Optimization in Deep Learning'. We reproduce strong baselines in time-series analysis, including the FCNs, RNNs (LSTM, GRU, LSTNet), CNNs \cite{TCN, WaveNet}, GNNs (GraphWaveNet) and attention models (TPA-LSTM). All the APIs are unified.
## As a Time-series Toolkit
XPBoost is a Easy-to-use, Modular and Extendible package of deep-learning based time-series models. You can use any complex model with model.fit()，and model.predict(). Facing a new scene, you can adapt the model with following steps:
- Define the task in the DataLoader object in the utils.py. The dominant problems in time-series, such as the monitoring, classification and forecasting, can be formulated via the dataloader definition.
- Select the model provided in the 'model' directory, or define a new model following the APIs in the provided models.
- The trainer has been defined in the root directory (in the class '\*Model'). If necessary, re-configure these trainers and hyper-parameters.

So far the following networks are supported :

|  Model   |  Paper  |
|  ----  | ----  |
| FCN  | [Generalization in fully-connected neural networks for 435 time series forecasting](https://www.sciencedirect.com/science/article/abs/pii/S1877750319301838) |
| FC-LSTM  | [Sequence to sequence learning with neural networks](https://arxiv.org/abs/1409.3215) |
| GRU  | [Learning phrase representations using RNN encoder–decoder for statistical machine translation](https://arxiv.org/abs/1406.1078) |
| LSTNet  | [Modeling long- and short-term temporal patterns with deep neural 431 networks](https://dl.acm.org/doi/abs/10.1145/3209978.3210006) |
| TCN  | [An empirical evaluation of generic convolutional and recurrent networks 445 for sequence modeling](https://arxiv.org/abs/1803.01271) |
| WaveNet  | [Wavenet: A generative model for raw audio](https://arxiv.org/abs/1609.03499) |
| GraphWaveNet  | [Graph wavenet for deep spatial-temporal graph modeling](https://arxiv.org/abs/1906.00121) |
| TPALSTM  | [Temporal pattern attention for multivariate time series forecasting](https://arxiv.org/abs/1809.04206) |

## As a Performance Booster
Further, XPBoost provides a method to boost the performance of deep learning models. Conduct it following the steps as:

- Train the deep learning models using the scripts provided as before.
- Run 'midGen.py' to generate the intermediate datasets.
- Run the meta-optimizers (PSO.py, CS.py, BAS.py) and view the enhanced results.

So far three meta-optimizers are supported by XPBoost:

|  Optimizer   |  Paper  |
|  ----  | ----  |
| Particle Swarm Optimization  | [A review on particle swarm optimization algorithm and its 350 variants to clustering high-dimensional data](https://dl.acm.org/doi/abs/10.1007/s10462-013-9400-4) |
| Cuckoo Search  | [Cuckoo search via levy flights](https://ieeexplore.ieee.org/abstract/document/5393690/) |
| Beetle Antennae Search  | [Bas: Beetle antennae search algorithm for optimization problems](https://arxiv.org/abs/1710.10724) |

## Disscussion
E-mail: 22032130@zju.edu.cn
