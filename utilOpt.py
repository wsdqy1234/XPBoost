import numpy as np
import os
import os.path as osp
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import mean_absolute_percentage_error


def evaluate(true, pred):
    """
    Calculate the metrics given the labels and predictions
    :param true: The ground-truth label value of all set
    :param pred: The predicted value of all set
    :return: The MAPE, MAE, MSE, R2
    """
    return mean_absolute_percentage_error(true, pred), mean_absolute_error(true, pred), mean_squared_error(true,
                                                                                                           pred), r2_score(
        true, pred)


def initPerformance(args, weight1, bias1, weight2, bias2, data, plsLatent):  # for ablation study
    """
    Calculate the metrics acquired by the initial trained network, the LR-enhanced network and the PLS-enhanced network
    :param args: The args in the main file
    :param weight1: The weight of the last-second layer
    :param bias1: The bias of the last-second layer
    :param weight2: The weight of the last layer
    :param bias2: The bias of the last layer
    :param data: The medium data generated using medGen.py, with keys {'trainX', 'valX', 'testX', 'trainY', 'valY', 'testY'}
    :param plsLatent: The latent number of PLS
    :return:
    """

    def lossInit(weight1, bias1, weight2, bias2, data, mode):  # SGD

        """
        Calculate the loss of the initial neural network trained using SGD for train, val and test set
        :param weight1: The weight of the last-second layer
        :param bias1: The bias of the last-second layer
        :param weight2: The weight of the last layer
        :param bias2: The bias of the last layer
        :param data: The medium data generated using medGen.py
        :param mode: 'train', 'val' or 'test'
        :return: The MAPE, MAE, MSE, R2 from evaluate function
        """

        a = np.matmul(data[mode + 'X'], weight1) + bias1
        a = np.tanh(a)
        pred = np.matmul(a, weight2) + bias2

        return evaluate(data[mode + 'Y'], pred)

    def lossLrInit(weight1, bias1, data, mode):  # SGD with Batch lr

        """
        Calculate the loss of the neural network enhanced using batch linear regression for train, val and test set
        :param weight1: The weight of the last-second layer
        :param bias1: The bias of the last-second layer
        :param data: The medium data generated using medGen.py
        :param mode: 'train', 'val' or 'test'
        :return: The MAPE, MAE, MSE, R2 from evaluate function
        """

        a = np.matmul(data['trainX'], weight1) + bias1
        a = np.tanh(a)
        lr = LinearRegression()
        lr.fit(a, data['trainY'])
        if mode == 'train':
            pred = lr.predict(a)
        else:
            a = np.matmul(data[mode + 'X'], weight1) + bias1
            a = np.tanh(a)
            pred = lr.predict(a)

        return evaluate(data[mode + 'Y'], pred)

    def lossPlsInit(weight1, bias1, data, mode):  # SGD with Batch lr

        """
        Calculate the loss of the neural network enhanced using batch PLSR for train, val and test set
        :param weight1: The weight of the last-second layer
        :param bias1: The bias of the last-second layer
        :param data: The medium data generated using medGen.py
        :param mode: 'train', 'val' or 'test'
        :return: The MAPE, MAE, MSE, R2 from evaluate function
        """
        a = np.matmul(data['trainX'], weight1) + bias1
        a = np.tanh(a)
        pls = PLSRegression(n_components=plsLatent)
        pls.fit(a, data['trainY'])
        if mode == 'train':
            pred = pls.predict(a)
        else:
            a = np.matmul(data[mode + 'X'], weight1) + bias1
            a = np.tanh(a)
            pred = pls.predict(a)

        return evaluate(data[mode + 'Y'], pred)

    path = osp.join('./works', args.dataName, args.modelName + args.taskName + str(args.horizon), str(args.seed),
                    '{0}initL{1}.txt'.format(args.optName, args.latent))

    if osp.exists(path):
        os.remove(path)

    with open(path, 'a') as f:

        mape, mae, rmse, r2 = lossInit(weight1, bias1, weight2, bias2, data, 'train')
        f.write('{0},{1},{2},{3}\n'.format(mape, mae, rmse, r2))
        mape, mae, rmse, r2 = lossInit(weight1, bias1, weight2, bias2, data, 'valid')
        f.write('{0},{1},{2},{3}\n'.format(mape, mae, rmse, r2))
        mape, mae, rmse, r2 = lossInit(weight1, bias1, weight2, bias2, data, 'test')
        f.write('{0},{1},{2},{3}\n'.format(mape, mae, rmse, r2))

        mape, mae, rmse, r2 = lossLrInit(weight1, bias1, data, 'train')
        f.write('{0},{1},{2},{3}\n'.format(mape, mae, rmse, r2))
        mape, mae, rmse, r2 = lossLrInit(weight1, bias1, data, 'valid')
        f.write('{0},{1},{2},{3}\n'.format(mape, mae, rmse, r2))
        mape, mae, rmse, r2 = lossLrInit(weight1, bias1, data, 'test')
        f.write('{0},{1},{2},{3}\n'.format(mape, mae, rmse, r2))

        mape, mae, rmse, r2 = lossPlsInit(weight1, bias1, data, 'train')
        f.write('{0},{1},{2},{3}\n'.format(mape, mae, rmse, r2))
        mape, mae, rmse, r2 = lossPlsInit(weight1, bias1, data, 'valid')
        f.write('{0},{1},{2},{3}\n'.format(mape, mae, rmse, r2))
        mape, mae, rmse, r2 = lossPlsInit(weight1, bias1, data, 'test')
        f.write('{0},{1},{2},{3}'.format(mape, mae, rmse, r2))


def cal_y(subVectors, plsLatent, shapeW1, data, mode):
    """
    Calculate the metrics for the subpopulation
    :param subVectors: The subpopulation. Each row include the flattened params including the [weight1, bias1, weight2, bias2]
    :param plsLatent: The latent number of PLS
    :param shapeW1: The shape of the weight1, to parse the subVectors
    :param data: The medium data generated using medGen.py
    :param mode: 'train', 'val' or 'test'
    :return: The metric list containing 4 metrics for all individuals in the subpopulation.
    """
    mapeList = []
    maeList = []
    rmseList = []
    r2List = []
    weight1Len = shapeW1[0] * shapeW1[1]

    for item in subVectors:

        weight1 = item[:weight1Len].reshape((shapeW1[0], shapeW1[1]))
        bias1 = item[weight1Len:].reshape((1, shapeW1[1]))
        a = np.matmul(data['trainX'], weight1) + bias1
        a = np.tanh(a)
        pls = PLSRegression(n_components=plsLatent)
        pls.fit(a, data['trainY'])

        if mode == 'train':
            predY = pls.predict(a).reshape(-1)

        if mode != 'train':
            a = np.matmul(data[mode + 'X'], weight1) + bias1
            a = np.tanh(a)
            predY = pls.predict(a).reshape(-1)

        mape, mae, rmse, r2 = evaluate(data[mode + 'Y'], predY)

        mapeList.append(mape)
        maeList.append(mae)
        rmseList.append(rmse)
        r2List.append(r2)

    return [mapeList, maeList, rmseList, r2List]


def record(opt, mode, plsLatent, shapeW1, data):
    """
    Record the metrics in the training process
    :param opt: The meta-optimizer object
    :param mode: 'train', 'val' or 'test'
    :param plsLatent: The latent number of PLS
    :param shapeW1: The shape of the weight1, to parse the subVectors
    :param data: The medium data generated using medGen.py
    :return: None
    """
    if mode == 'best':
        with open(opt.logBestDir['train'], 'a') as f:
            mape, mae, rmse, r2 = cal_y([opt.best['location']], plsLatent, shapeW1, data, mode='train')
            tempTrain = np.array([opt.iter, mape[0], mae[0], rmse[0], r2[0]])
            tempTrain = np.concatenate((tempTrain, opt.best['location']))
            np.savetxt(f, tempTrain.reshape(1, -1), delimiter=',', fmt='%s')
            print('epoch={0}, rmse={1}, r2={2}'.format(int(opt.iter), rmse[0], r2[0]))

        with open(opt.logBestDir['valid'], 'a') as f:
            mape, mae, rmse, r2 = cal_y([opt.best['location']], plsLatent, shapeW1, data, mode='valid')
            f.write('{0},{1},{2},{3},{4}\n'.format(opt.iter, mape[0], mae[0], rmse[0], r2[0]))
            print('epoch={0}, rmse={1}, r2={2}'.format(int(opt.iter), rmse[0], r2[0]))

        with open(opt.logBestDir['test'], 'a') as f:
            mape, mae, rmse, r2 = cal_y([opt.best['location']], plsLatent, shapeW1, data, mode='test')
            f.write('{0},{1},{2},{3},{4}\n'.format(opt.iter, mape[0], mae[0], rmse[0], r2[0]))
            print('epoch={0}, rmse={1}, r2={2}'.format(int(opt.iter), rmse[0], r2[0]))

    if mode == 'ordinary':
        with open(opt.logOrdinaryDir['train'], 'a') as f:
            np.savetxt(f, opt.loss.reshape(1, -1), delimiter=',', fmt='%s')
            # f.write('\n')
