# XPBoost
This is the implementation of the paper 'eXtreme Performance Boosting: Revisiting Statistics and Intelligent Optimization in Deep Learning'. We reproduce 8 strong baselines in time-series analysis, including the FCNs, RNNs (lstm, gru, lstnet), CNNs \cite{tcn, wavenet}, GNNs (GraphWaveNet) and attention models (TPA-LSTM). All the APIs are unified.

You can utilize this as a toolkit to reproduce the papers in time-series domain, or acquire a demo with low development cost. Facing a new scene, you can adapt the model with following steps:
- Define the task in the DataLoader object in the utils.py. The dominant problems in time-series domain, ranging from the monitoring, classification to forecasting, can be formulated via the dataloader definition.
- Select the model provided in the 'model' directory, or define a new model following the APIs in the provided models.
- The trainer has been defined in the root directory (in the class '*Model'). If necessary, re-configure these trainers and hyper-parameters.

So far the following networks are supported :

- FCN: It captures the non-linear relationship between input and output variables. In our experiments, we select the 4-layer FCN with the hidden size 64, 64, 64, 1. There is no difficulty of reproduction, readers can refer to our repository directly for specific implement directly. Other settings follow the default values in the paper. 
- FC-LSTM: It encodes the non-linear auto-relationship and correlationship between the input variables and output variables simultaneously in hidden space. The source code can be found at \url{https://github.com/farizrahman4u/seq2seq}. In our experiments, we utilize the LSTM cell of 100 hidden size. Other settings follow the recommended values in the paper. 
- GRU: It forecasts univariate time-series with fully-connected GRU hidden units, which is a variant of FC-LSTM. The source code can be found at \url{https://github.com/zhangxu0307/time_series_forecasting_pytorch}. In our experiments, we utilize the GRU cell of 100 hidden size.
- LSTNet: It takes advantage of the convolution layer to discover the local dependence patterns among multi-dimensional input variables, and the recurrent layer to capture the long-term dependency patterns. We use the open source code from \url{https://github.com/fbadine/LSTNet}. In our experiments, the number of output filters in the CNN layer is 100, the latent size of the RNN layer is 100, and the CNN filter size is 6. Other experimental settings follow the default values in the paper.
- TCN: It combines the best practices such as the dilations convolution and causal convolution for autoregressive prediction. We take the source code at \url{https://github.com/1ocuslab/TCN}. In our experiments, the kernel size is 2, the numbers of output channels are 32, 64, 128, 128. Other configurations such as the dynamic dilation value follow the default values in the paper. 
- WaveNet: It introduces the skip and residual connection into TCNs, for multi-scale information fusion and fast convergence. We take the source code at \url{https://github.com/LongxingTan/Time-series-prediction}. In experiments, the layer size is 7, the skip size is 1, the numbers of output channels, residual channels and skip channels are 100. Other settings follow the recommended values in the paper. 
- GraphWaveNet: It encodes each node's neighborhood via a low-dimensional embedding by leveraging heat wavelet diffusion patterns. We take the source code at \url{ https://github.com/nnzhan/Graph-WaveNet}.  In our experiments, the number of output channels, residual channels, skip channels and dilation channels are 32, the kernel size is 2, the layer size is 7 and the skip size is 1. we turn on the ‘aptonly’ and 'addaptadj' options to generate adaptive graphs for all datasets. Other configurations follow the options recommended in the paper.
- TPALSTM: It introduces a set of filters to extract time-invariant temporal patterns, which is similar to transform data from time domain into frequency domain. We take the source code at \url{https://github.com/gantheory/TPA-LSTM}. In our experiments, the latent size of the LSTM layer is 100, the number of convolution filter is 100, the kernel size of convolution filter is 6. Other configurations such as the dynamic dilation value follow the default values in the paper.

Further, XPBoost provides a method to boost the performance of deep learning models. Conduct it following the steps as:

- Train the deep learning models using the scripts provided as before.
- Run 'midGen.py' to generate the intermediate datasets.
- Run the optimizers (PSO.py, CS.py, BAS.py) and view the enhanced results.
