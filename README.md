# XPBoost
This is the implementation of the paper 'eXtreme Performance Boosting: Revisiting Statistics and Intelligent Optimization in Deep Learning'. We reproduce 8 strong baselines in time-series analysis, including the FCNs, RNNs (LSTM, GRU, LSTNet), CNNs \cite{TCN, WaveNet}, GNNs (GraphWaveNet) and attention models (TPA-LSTM). All the APIs are unified.
## Use as Toolkits
XPBoost is a Easy-to-use, Modular and Extendible package of deep-learning based time-series models. You can use any complex model with model.fit()，and model.predict(). Facing a new scene, you can adapt the model with following steps:
- Define the task in the DataLoader object in the utils.py. The dominant problems in time-series, such as the monitoring, classification and forecasting, can be formulated via the dataloader definition.
- Select the model provided in the 'model' directory, or define a new model following the APIs in the provided models.
- The trainer has been defined in the root directory (in the class '\*Model'). If necessary, re-configure these trainers and hyper-parameters.

So far the following networks are supported :

|  Model   |  Paper  |
|  ----  | ----  |
| FCN  | Generalization in fully-connected neural networks for 435 time series forecasting |
| FC-LSTM  | Sequence to sequence learning with neural networks |
| GRU  | Learning phrase representations using RNN encoder–decoder for statistical machine translation |
| LSTNet  | Modeling long- and short-term temporal patterns with deep neural 431 networks |
| TCN  | An empirical evaluation of generic convolutional and recurrent networks 445 for sequence modeling |
| WaveNet  | Wavenet: A generative model for raw audio |
| GraphWaveNet  | Graph wavenet for deep spatial-temporal graph modeling |
| TPALSTM  | Temporal pattern attention for multivariate time series forecasting |

## Use for Performance Improvement
Further, XPBoost provides a method to boost the performance of deep learning models. Conduct it following the steps as:

- Train the deep learning models using the scripts provided as before.
- Run 'midGen.py' to generate the intermediate datasets.
- Run the meta-optimizers (PSO.py, CS.py, BAS.py) and view the enhanced results.

So far three meta-optimizers are supported by XPBoost:
- Particle Swarm Optimization \cite{pso}: The Particle Swarm Optimization (PSO) utilizes a population of candidate particles, of which the positions are denoted as solutions, to solve optimization problems. These particles are moving in the search-space with a specific velocity, which is updated based on its local best position and global best position among all particles. PSO consists of three important parameters: the momentum, the factor of the global-wise optimum, and the factor of particle-wise-optimum. The source code can be found from \url{https://github.com/ljvmiranda921/pyswarms} under MIT license. With the collaboration and information sharing between individuals in the population, PSO is possible to help neural network jump out of local optima and approximate the global optimal solution.

- Cuckoo Search \cite{cs}: The Cuckoo Search (CS) algorithm solves optimization problems by simulating the brood parasitism of cuckoos. Specifically, Some species of cockoos dumps their eggs in nests of other host birds of different species. This algorithm uses eggs in nests as a representation of solutions, which are updated based on three steps: a) Update the position of cuckoos based on the best nest and step factor, b) Conduct Levy flight and update the position of nests. c) With a certain possibility discover cuckoo eggs and rebuild the nest by host birds, while the number of available nests is maintained. The source code can be found from \url{https://github.com/ashwinwagh96/SNN-model-using-Cuckoo-Search-Algorithm}.

- Beetle Antennae Search \cite{bas}: The Beetle Antennae Search (BAS) is an algorithm developed based on the behavior of beetle foraging involving antennae searching and random walking. The bettle receive the odours of prey, i.e. the function value, with two tentacles in opposite positions, and tends to move in the direction with a higher concentration of odour. BAS only requires one individual during search and thus greatly simplify calculation compared to population-based algorithms. However, to exploit the potential of this algorithm and make the results comparable, we initialize N beetle antennaes and optimize them simultaneously. In our experiments, the distance between the two antennae $d_0=0.2$, $\eta=0.95$. The source code can be found from \url{https://github.com/AAFun/pybas} under GPL-3.0 License.

## Disscussion
E-mail: 22032130@zju.edu.cn
