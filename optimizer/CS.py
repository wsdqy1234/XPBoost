import numpy as np
import scipy.special as sc_special
import argparse
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error


class CuSearch():
    """
    This is the original implementation of the Cuckoo search algorithm
    """

    def __init__(self, args, init):

        self.pop = args.csPop
        self.dim = init.shape[1]
        self.Lb = args.csLb * np.ones(shape=(self.pop, self.dim))
        self.Ub = args.csUb * np.ones(shape=(self.pop, self.dim))
        self.drop = args.csDrop
        self.beta = args.csBeta
        self.step = args.csStep

        self.nest = self.Lb + (self.Ub - self.Lb) * np.random.randn(self.pop, self.dim)
        self.nest[0] = init  # 最优
        self.nestTemp = None
        self.Y = None
        self.YTemp = None
        self.bestNest = {'X': None, 'Y': None}
        self.iter = 0

    def updateBestNest(self):

        place = np.argmax(self.Y)
        if self.iter == 0:
            self.bestNest['X'] = self.nest[place]
            self.bestNest['Y'] = self.Y[place]
        else:
            if self.Y[place] > self.bestNest['Y']:
                self.bestNest['X'] = self.nest[place]
                self.bestNest['Y'] = self.Y[place]

    def updateNest(self):

        steps = levyFlight(self.beta, self.nest.shape)
        self.nestTemp = self.nest.copy()
        stepSize = self.step * steps * (self.nest - self.bestNest['X'])
        stepDirect = np.random.rand(self.nest.shape[0], self.nest.shape[1])
        self.nestTemp += stepSize * stepDirect
        self.nestTemp = np.clip(self.nestTemp, self.Lb, self.Ub)

    def abandonNest(self):

        n, m = self.nest.shape
        for idx in range(n):
            if (np.random.rand() < self.drop):
                self.nest[idx] += np.random.rand() * (
                        self.nest[np.random.randint(0, n)] - self.nest[np.random.randint(0, n)])
        self.nest = np.clip(self.nest, self.Lb, self.Ub)


def levyFlight(beta, shape):
    sigmaU = (sc_special.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
            sc_special.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
    sigmaV = 1
    u = np.random.normal(0, sigmaU, shape)
    v = np.random.normal(0, sigmaV, shape)
    steps = u / ((np.abs(v)) ** (1 / beta))

    return steps


def cal_y(X):
    Y = np.ones(shape=(len(X)))
    for i, item in enumerate(X):
        x, y = item
        Y[i] = 3 * (1 - x) ** 2 * np.e ** (-x ** 2 - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.e ** (
                -x ** 2 - y ** 2) - (np.e ** (-(x + 1) ** 2 - y ** 2)) / 3
    return Y


def cal_y(plsLatent, subVectors, shapeW1, data):
    X = []
    trainLoss = []
    trainR2 = []
    weight1Len = shapeW1[0] * shapeW1[1]

    for item in subVectors:
        weight1 = item[:weight1Len].reshape((shapeW1[0], shapeW1[1]))
        bias1 = item[weight1Len:].reshape((1, shapeW1[1]))
        a = np.tanh(np.matmul(data['trainX'], weight1) + bias1)

        pls = PLSRegression(n_components=plsLatent)
        # pls = LinearRegression()
        pls.fit(a, data['trainY'])
        predY = pls.predict(data['trainX'])
        train_loss = np.sqrt(mean_squared_error(data['trainY'], predY))
        r2 = r2_score(data['trainY'], predY)

        X.append(item)
        trainLoss.append(train_loss)
        trainR2.append(r2)

    return [X, trainLoss, trainR2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='None')
    parser.add_argument('--csPop', type=int, default=2000)
    parser.add_argument('--csLb', type=float, default=-3)
    parser.add_argument('--csUb', type=float, default=3)
    parser.add_argument('--csIter', type=int, default=100)
    parser.add_argument('--csDrop', type=float, default=0.25)
    parser.add_argument('--csBeta', type=float, default=1.5)
    parser.add_argument('--csStep', type=float, default=0.01)

    args = parser.parse_args()
    cs = CuSearch(args, init=np.ones((1, 2)))

    plsLatent = 48
    numPerProcess = 10
    while cs.iter < args.csIter:

        for idx in range(int(args.csPop // numPerProcess)):
            subVectors = cs.nest[idx * numPerProcess: (idx + 1) * numPerProcess]
            cs.Y = cal_y(plsLatent, subVectors, )

        cs.updateBestNest()

        cs.updateNest()
        YTemp = cal_y(cs.nestTemp)
        cs.nest[YTemp > cs.Y] = cs.nestTemp[YTemp > cs.Y]

        cs.abandonNest()
        cs.iter += 1
    print(cs.bestNest)
