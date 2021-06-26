import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from Optim import Optim
import argparse
import pandas as pd
from sklearn.metrics import mean_absolute_error, explained_variance_score, mean_squared_error, median_absolute_error, \
    r2_score
from utils import DataS, DataM, mean_absolute_percentage_error
from models.TCN import TCN
from utils import mkdir
import os.path as osp


class TCNModel(BaseEstimator, RegressorMixin):

    def __init__(self, args):

        super(TCNModel, self).__init__()

        torch.manual_seed(args.seed)

        self.epoch = 1
        self.epochs = args.epochs
        self.batchSize = args.batchSize
        self.lr = args.lr

        if args.taskName == 'S':

            self.trainSet = DataS(args, mode='Train')
            self.validSet = DataS(args, mode='Val', scaler=self.trainSet.scaler)
            self.dimI = self.trainSet.data.shape[1] - 1
            self.dimO = 1

        else:

            self.trainSet = DataM(args, mode='Train')
            self.validSet = DataM(args, mode='Val', scaler=self.trainSet.scaler)
            self.dimI = self.trainSet.data.shape[1]
            self.dimO = self.dimI

        self.criterion = nn.MSELoss()

        if args.modelName == 'TCN':
            self.model = TCN(args, self.dimI, self.dimO).to(torch.device('cuda:{}'.format(args.device)))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)

        self.trainDict = dict()
        self.evalDict = dict()
        self.trainDict['loss'] = np.array([])
        self.evalDict['medianAbsoluteError'] = np.array([])
        self.evalDict['explainedVarianceScore'] = np.array([])
        self.evalDict['meanAbsoluteError'] = np.array([])
        self.evalDict['meanAbsolutePercentageError'] = np.array([])
        self.evalDict['rootMeanSquaredError'] = np.array([])
        self.evalDict['r2'] = np.array([])
        self.evalDict['r2Best'] = 0
        self.savePath = osp.join('./works', args.dataName, args.modelName + args.taskName + str(args.horizon),
                                 str(args.seed))
        mkdir(self.savePath)

    def fit(self):

        trainLoader = DataLoader(dataset=self.trainSet, batch_size=self.batchSize, shuffle=True)

        for i in range(1, self.epochs + 1):

            self.model.train()
            self.epoch = i

            for batchX, batchY in trainLoader:
                self.model.zero_grad()
                output = self.model(batchX)
                output = output.reshape(batchY.shape)
                loss = self.criterion(output, batchY)  # shape
                loss.backward()
                self.optimizer.step()
                self.trainDict['loss'] = np.append(self.trainDict['loss'], loss.data.cpu().numpy())

            # self.scheduler.step()

            print('Epoch: {}, Loss: {}'.format(self.epoch, loss.data.cpu().numpy()))

            self.valid()
            if self.epoch % 20 == 0:
                torch.save(model, osp.join(self.savePath, 'epoch{}.pth'.format(self.epoch)))

            if self.evalDict['r2'][-1].mean() > self.evalDict['r2Best']:
                self.evalDict['r2Best'] = self.evalDict['r2'][-1].mean()

                if self.epoch > 10:
                    torch.save(model, osp.join(self.savePath, 'best.pth'.format(self.epoch)))

        pd.DataFrame(self.trainDict).to_csv(self.savePath + '/trainLog.txt', header=0, index=0)
        pd.DataFrame(self.evalDict).to_csv(self.savePath + '/validLog.txt', header=0, index=0)

    def valid(self):

        with torch.no_grad():
            self.model.eval()
            validLoader = DataLoader(dataset=self.validSet, batch_size=self.batchSize, shuffle=True)
            results = {'output': np.array([]), 'label': np.array([])}

            for batchX, batchY in validLoader:
                output = self.model(batchX)
                output = output.reshape(batchY.shape).data.cpu().numpy()
                batchY = batchY.data.cpu().numpy()
                results['label'] = np.append(results['label'], batchY)
                results['output'] = np.append(results['output'], output)

            r2 = r2_score(results['label'], results['output'])
            explainedVarianceScore = explained_variance_score(results['label'], results['output'])
            rootMeanSquaredError = np.sqrt(mean_squared_error(results['label'], results['output']))
            meanAbsoluteError = mean_absolute_error(results['label'], results['output'])
            medianAbsoluteError = median_absolute_error(results['label'], results['output'])
            meanAbsolutePercentageError = mean_absolute_percentage_error(results['label'], results['output'])

            self.evalDict['r2'] = np.append(self.evalDict['r2'], r2)
            self.evalDict['explainedVarianceScore'] = np.append(self.evalDict['explainedVarianceScore'],
                                                                explainedVarianceScore)
            self.evalDict['rootMeanSquaredError'] = np.append(self.evalDict['rootMeanSquaredError'],
                                                              rootMeanSquaredError)
            self.evalDict['meanAbsoluteError'] = np.append(self.evalDict['meanAbsoluteError'], meanAbsoluteError)
            self.evalDict['medianAbsoluteError'] = np.append(self.evalDict['medianAbsoluteError'], medianAbsoluteError)
            self.evalDict['meanAbsolutePercentageError'] = np.append(self.evalDict['meanAbsolutePercentageError'],
                                                                     meanAbsolutePercentageError)

        print('Epoch: {}, Valid Loss: {}, R2: {}'.format(self.epoch, rootMeanSquaredError, r2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    parser.add_argument('--dataName', type=str, default='traffic')
    parser.add_argument('--taskName', type=str, default='S')
    parser.add_argument('--modelName', type=str, default='TCN')
    parser.add_argument('--dimFC', type=int, default=64, help='number of FC units')

    parser.add_argument('--num_channels', default=[32, 64, 128, 128])
    parser.add_argument('--kernelSize', default=2)

    parser.add_argument('--window', type=int, default=24 * 7, help='window size')
    parser.add_argument('--highway_window', type=int, default=0, help='The window size of the highway component')
    parser.add_argument('--clip', type=float, default=10., help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batchSize', type=int, default=32, metavar='N')
    parser.add_argument('--dropout', type=float, default=0.1, help='(0 = no dropout)')
    parser.add_argument('--device', default=7)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1024)
    args = parser.parse_args()

    model = TCNModel(args)
    model.fit()
    model.valid()
