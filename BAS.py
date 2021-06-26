import numpy as np
from numpy.core.records import record
import scipy.special as sc_special
import argparse
import torch
import os
import os.path as osp
import numpy as np
import argparse
from LSTNet import LSTNetModel
from DCTCN import DCTCNModel
from TCN import TCNModel
from DNN import DNNModel
from TPALSTM import TPALSTMModel
from WaveNet import WaveNetModel
from GraphWaveNet import GraphWaveNetModel
from joblib import Parallel, parallel_backend, delayed
from utilOpt import initPerformance, cal_y, record


class PSO():

    def __init__(self, args, init):

        self.pop = args.optPop
        self.dim = init.shape[1]
        self.lb = args.optLb * np.ones(shape=(self.pop, self.dim))
        self.ub = args.optUb * np.ones(shape=(self.pop, self.dim))
        self.eta = args.optEta
        self.c = args.optC
        self.step = args.optStep
        self.constrain = True

        self.location = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        self.location[0] = init  # 最优
        self.locationL = None
        self.locationR = None
        self.loss = None
        self.best = {'location': init, 'loss': -np.inf}
        self.iter = int(0)

        logDir = osp.join('./works', args.dataName, args.modelName + args.taskName + str(args.horizon), str(args.seed))
        self.logBestDir = {
            'train': osp.join(logDir, '{0}L{1}P{2}BestTrain.txt'.format(args.optName, args.latent, self.pop)),
            'valid': osp.join(logDir, '{0}L{1}P{2}BestValid.txt'.format(args.optName, args.latent, self.pop)),
            'test': osp.join(logDir, '{0}L{1}P{2}BestTest.txt').format(args.optName, args.latent, self.pop)}
        self.logOrdinaryDir = {
            'train': osp.join(logDir, '{0}L{1}P{2}Ordinary.txt'.format(args.optName, args.latent, self.pop))}
        [os.remove(item) for item in self.logBestDir.values() if osp.exists(item)]
        [os.remove(item) for item in self.logOrdinaryDir.values() if osp.exists(item)]

    def updateState(self):

        d0 = self.step / self.c
        dir = np.random.randn(self.pop, self.dim)
        dir = dir / (10e-5 + np.linalg.norm(dir, axis=1, keepdims=True))
        self.locationL = self.location - dir * d0
        self.locationR = self.location + dir * d0

    def updateBest(self):

        place = np.argmax(self.loss)
        if self.iter == 0:
            self.best['location'] = self.location[place]
            self.best['loss'] = self.loss[place]
            return True
        else:
            if self.loss[place] > self.best['loss']:
                self.best['location'] = self.location[place]
                self.best['loss'] = self.loss[place]
                return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='optimizer')

    parser.add_argument('--optName', type=str, default='bas')
    parser.add_argument('--dataName', type=str, default='traffic')
    parser.add_argument('--taskName', type=str, default='S')
    parser.add_argument('--modelName', type=str, default='LSTM')
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--latent', type=int, default=48)

    parser.add_argument('--optPop', type=int, default=100)
    parser.add_argument('--optLb', type=float, default=-1)
    parser.add_argument('--optUb', type=float, default=1)
    parser.add_argument('--optIter', type=int, default=100)
    parser.add_argument('--optC', type=float, default=5)
    parser.add_argument('--optEta', type=float, default=0.95)
    parser.add_argument('--optStep', type=float, default=1)

    parser.add_argument('--numPerProcess', type=int, default=10)
    args = parser.parse_args()

    modelDir = osp.join('./works', args.dataName, args.modelName + args.taskName + str(args.horizon), str(args.seed))
    model = torch.load(osp.join(modelDir, 'best.pth')).model

    pretrained_dict = model.cpu().state_dict()

    weight1 = pretrained_dict['FC1.weight'].numpy()
    weight1 = weight1.transpose(1, 0)
    weight1Vector = weight1.reshape(1, -1)

    bias1 = pretrained_dict['FC1.bias'].numpy().reshape(1, -1)
    bias1Vector = bias1

    weight2 = pretrained_dict['FC2.weight'].numpy()
    weight2 = weight2.transpose(1, 0)
    bias2 = pretrained_dict['FC2.bias'].numpy().reshape(1, -1)

    print('weight1.shape = {0}, bias1.shape = {1}, weight2.shape = {2}, bias2.shape = {3}'.format(weight1.shape,
                                                                                                  bias1.shape,
                                                                                                  weight2.shape,
                                                                                                  bias2.shape))
    initVector = np.concatenate([weight1Vector, bias1Vector], axis=1)
    print(initVector.shape)

    data = {}
    dataTrain = np.loadtxt(os.path.join(modelDir, 'medTrain.csv'), delimiter=',')
    dataValid = np.loadtxt(os.path.join(modelDir, 'medValid.csv'), delimiter=',')
    dataTest = np.loadtxt(os.path.join(modelDir, 'medTest.csv'), delimiter=',')
    data['trainX'] = dataTrain[:, :-1]
    data['trainY'] = dataTrain[:, -1]
    data['validX'] = dataValid[:, :-1]
    data['validY'] = dataValid[:, -1]
    data['testX'] = dataTest[:, :-1]
    data['testY'] = dataTest[:, -1]

    opt = PSO(args, init=initVector)

    initPerformance(args, weight1, bias1, weight2, bias2, data, args.latent)

    while opt.iter < args.optIter:

        with parallel_backend('multiprocessing', n_jobs=10):

            results = Parallel()(
                delayed(cal_y)(opt.location[idx * args.numPerProcess: (idx + 1) * args.numPerProcess], args.latent,
                               weight1.shape, data, 'train') for idx in range(int(args.optPop // args.numPerProcess)))

            mape = [item[0] for item in results]
            mae = [item[1] for item in results]
            rmse = [item[2] for item in results]
            r2 = [item[3] for item in results]
            opt.loss = np.array(r2).reshape(-1)

        recorder = opt.updateBest()

        if recorder == True:
            record(opt, 'best', args.latent, weight1.shape, data)

        opt.updateState()

        with parallel_backend('multiprocessing', n_jobs=10):

            results = Parallel()(
                delayed(cal_y)(opt.locationL[idx * args.numPerProcess: (idx + 1) * args.numPerProcess], args.latent,
                               weight1.shape, data, 'train') for idx in range(int(args.optPop // args.numPerProcess)))

            mape = [item[0] for item in results]
            mae = [item[1] for item in results]
            rmse = [item[2] for item in results]
            r2 = [item[3] for item in results]
            lossL = np.array(r2).reshape(-1)
        with parallel_backend('multiprocessing', n_jobs=10):

            results = Parallel()(
                delayed(cal_y)(opt.locationR[idx * args.numPerProcess: (idx + 1) * args.numPerProcess], args.latent,
                               weight1.shape, data, 'train') for idx in range(int(args.optPop // args.numPerProcess)))

            mape = [item[0] for item in results]
            mae = [item[1] for item in results]
            rmse = [item[2] for item in results]
            r2 = [item[3] for item in results]
            lossR = np.array(r2).reshape(-1)
        # lossL = cal_y([opt.locationL], args.latent, weight1.shape, data, 'train')
        # lossR = cal_y([opt.locationR], args.latent, weight1.shape, data, 'train')
        opt.location[lossL < lossR] = opt.locationL[lossL < lossR]
        opt.location[lossL > lossR] = opt.locationR[lossL > lossR]
        record(opt, 'ordinary', args.latent, weight1.shape, data)
        opt.iter += 1
        print('{0} iter={1}'.format(args.optName, opt.iter))
