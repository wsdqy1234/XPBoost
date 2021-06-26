import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Iterable
import os
import os.path as osp
from utils import DataS, DataM
import numpy as np
import pandas as pd
import argparse
from LSTNet import LSTNetModel
from DCTCN import DCTCNModel
from TCN import TCNModel
from DNN import DNNModel
from TPALSTM import TPALSTMModel
from WaveNet import WaveNetModel
from GraphWaveNet import GraphWaveNetModel


class FeatureExtractor(nn.Module):
    '''
    extract the input of layer in given model
    '''

    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self._features = {layer: None for layer in layers}
        for idx in layers:
            layer = dict([*self.model.named_children()])[idx]
            layer.register_forward_hook(self.save_outputs_hook(idx))

    def save_outputs_hook(self, idx: str):
        def fn(module, input, output):
            self._features[idx] = input

        return fn

    def forward(self, x):
        self.model.eval()
        _ = self.model(x)
        return self._features


def extract(args):
    modelDir = osp.join('./works', args.dataName, args.modelName + args.taskName + str(args.horizon), str(args.seed))
    model = torch.load(osp.join(modelDir, 'best.pth'), map_location='cuda:{0}'.format(args.device)).model

    if args.taskName == 'S':
        trainSet = DataS(args, mode='Train')
        validSet = DataS(args, mode='Val', scaler=trainSet.scaler)
        testSet = DataS(args, mode='Test', scaler=trainSet.scaler)
    else:
        trainSet = DataM(args, mode='Train')
        validSet = DataM(args, mode='Val', scaler=trainSet.scaler)
        testSet = DataM(args, mode='Test', scaler=trainSet.scaler)

    trainLoader = DataLoader(dataset=trainSet, batch_size=args.batchSize)
    validLoader = DataLoader(dataset=validSet, batch_size=args.batchSize)
    testLoader = DataLoader(dataset=testSet, batch_size=args.batchSize)
    extractor = FeatureExtractor(model, ['FC1'])

    model.eval()
    model = model.to(torch.device('cuda:{0}'.format(args.device)))

    with torch.no_grad():

        for idx, dataloader in enumerate([trainLoader, validLoader, testLoader]):

            if idx == 0:
                f = osp.join(modelDir, "medTrain.csv")
            if idx == 1:
                f = osp.join(modelDir, "medValid.csv")
            if idx == 2:
                f = osp.join(modelDir, "medTest.csv")
            if osp.exists(f):
                os.remove(f)

            for batch_index, (input, labels) in enumerate(dataloader):
                # input = Variable(input).cuda()
                features = extractor(input)
                fc1 = features['FC1'][0].cpu().numpy()
                label = labels.cpu().numpy().reshape((-1, 1))
                output = np.concatenate((fc1, label), axis=1)
                pd.DataFrame(output).to_csv(f, header=None, index=None, mode='a')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    parser.add_argument('--modelName', type=str, default='LSTM')
    parser.add_argument('--dataName', default='solar_AL')
    parser.add_argument('--taskName', type=str, default='S')

    parser.add_argument('--window', type=int, default=24 * 7, help='window size')
    parser.add_argument('--batchSize', type=int, default=32, metavar='N')
    parser.add_argument('--device', default=7)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1024)
    args = parser.parse_args()
    print(args.dataName, args.modelName)
    extract(args)
    print('success!')
