import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


def mkdir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


class DataS(Dataset):
    """
    For the single-step single-output forecasting task
    x_{t-args.window:t} -> y_{t+args.horizon}
    """

    def __init__(self, args, mode, scaler=None):
        """
        :param args: The args defined in the main file
        :param mode: 'Train' and others
        :param scaler: If the mode is not 'Train', set the scaler object here.
        """

        self.mode = mode
        self.window = args.window
        self.h = args.horizon
        self.data = pd.read_csv('./data/{0}{1}.txt'.format(args.dataName, mode), header=None, index_col=None).values
        self.data = np.concatenate((self.data[1:, :-1], self.data[0:-1, [-1]], self.data[1:, [-1]]), axis=1)

        self.n, self.featureNum = self.data.shape
        self.featureNum = self.featureNum - 1  # 让m是输入特征维数
        self._normalized(scaler)
        self.data = torch.tensor(self.data, dtype=torch.float32, device=torch.device('cuda:{}'.format(args.device)))

    def _normalized(self, scaler):
        """
        :param scaler: If self.mode == 'Train', then the scaler is not used.  Otherwise, the scaler should keep consisent with the training data
        :return: None
        """

        if self.mode == 'Train':
            self.scaler = MinMaxScaler()
            self.data = self.scaler.fit_transform(self.data)
        else:
            self.scaler = scaler
            self.data = self.scaler.transform(self.data)

    def __getitem__(self, index):

        start = index
        end = index + self.window
        return self.data[start:end, :-1], self.data[end - 1, -1]

    def __len__(self):
        """
        The length of the data generated is related to the horizon and windowSize.
        :return: The length of preprocessed data
        """

        return len(self.data) - self.window - self.h


class DataM(Dataset):
    """
    For the single-step multi-output forecasting task
    x_{t-args.window:t} -> x_{t+args.horizon}
    """

    def __init__(self, args, mode, scaler=None):
        """
        :param args: The args defined in the main file
        :param mode: 'Train' and others
        :param scaler: If the mode is not 'Train', set the scaler object here.
        """

        self.mode = mode
        self.window = args.window
        self.h = args.horizon
        self.data = pd.read_csv('./data/{}.txt'.format(args.dataName), header=None, index_col=None).values
        self.data = self._split(int(args.trainProb * len(self.data)),
                                int((args.trainProb + args.validProb) * len(self.data)))
        self.n, self.featureNum = self.data.shape
        self._normalized(scaler);
        self.data = torch.tensor(self.data, dtype=torch.float32, device=torch.device('cuda:{}'.format(args.device)))

    def _split(self, train_num, valid_num):

        if self.mode == 'train':
            return self.data[:train_num, :]
        if self.mode == 'valid':
            return self.data[train_num:valid_num, :]
        if self.mode == 'test':
            return self.data[valid_num:, :]

    def _normalized(self, scaler):
        """
        :param scaler: If self.mode == 'Train', then the scaler is not used.  Otherwise, the scaler should keep consisent with the training data
        :return: None
        """

        if self.mode == 'train':
            self.scaler = MinMaxScaler()
            self.data = self.scaler.fit_transform(self.data)
        else:
            self.scaler = scaler
            self.data = self.scaler.transform(self.data)

    def __getitem__(self, index):

        start = index
        end = index + self.window

        return self.data[start:end], self.data[end]

    def __len__(self):
        """
        The length of the data generated is related to the horizon and windowSize.
        :return: The length of preprocessed data
        """

        return len(self.data) - self.window - self.h


def mean_absolute_percentage_error(true, pred):
    """
    Calculate the MAPE (The sklearn package does not provide this important metric)
    :param true: The ground-truth label of the total set
    :param pred: The predicted value of the total set
    :return: MAPE
    """
    for i in range(len(true)):
        if true[i] == 0.00:
            true[i] = 0.0001
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)
