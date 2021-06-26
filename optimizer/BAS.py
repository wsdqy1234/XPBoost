import numpy as np
import argparse


class BASearch():
    """
    This is the original implementation of the Beetle Antennae search algorithm
    """

    def __init__(self, args, init) -> None:

        self.pop = args.basPop
        self.dim = init.shape[1]
        self.Lb = args.basLb * np.ones(shape=(self.pop, self.dim))
        self.Ub = args.basUb * np.ones(shape=(self.pop, self.dim))

        self.eta = 0.95
        self.c = 5  # ratio between step and d0
        self.step = 1  # larges input range

        self.X = self.Lb + (self.Ub - self.Lb) * np.random.randn(self.pop, self.dim)
        self.leftX = None
        self.rightX = None
        self.X[0] = init  # 最优

        self.Y = None

        self.best = {'X': None, 'Y': None}
        self.iter = 0

    def updateBest(self):

        place = np.argmax(self.Y)
        if self.iter == 0:
            self.best['X'] = self.X[place]
            self.best['Y'] = self.Y[place]
        else:
            if self.Y[place] > self.best['Y']:
                self.best['X'] = self.X[place]
                self.best['Y'] = self.Y[place]

    def stateCal(self):

        d0 = self.step / self.c
        dir = np.random.randn(self.pop, self.dim)
        dir = dir / (10e-5 + np.linalg.norm(dir, axis=1, keepdims=True))
        self.leftX = self.X + dir * d0
        self.rightX = self.X - dir * d0


def cal_y(X):
    Y = np.ones(shape=(len(X)))
    for i, item in enumerate(X):
        x, y = item
        Y[i] = 3 * (1 - x) ** 2 * np.e ** (-x ** 2 - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.e ** (
                    -x ** 2 - y ** 2) - (np.e ** (-(x + 1) ** 2 - y ** 2)) / 3
    return Y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='None')
    parser.add_argument('--basPop', type=int, default=2)
    parser.add_argument('--basLb', type=float, default=-3)
    parser.add_argument('--basUb', type=float, default=3)
    parser.add_argument('--basIter', type=int, default=100)
    parser.add_argument('--basDrop', type=float, default=0.25)
    parser.add_argument('--basBeta', type=float, default=1.5)
    parser.add_argument('--basStep', type=float, default=0.01)

    args = parser.parse_args()
    bas = BASearch(args, init=np.ones((1, 2)))

    while bas.iter < args.basIter:
        bas.Y = cal_y(bas.X)
        bas.updateBest()
        bas.stateCal()

        leftY = cal_y(bas.leftX)
        rightY = cal_y(bas.rightX)
        bas.X[leftY > rightY] = bas.leftX[leftY > rightY]
        bas.X[leftY < rightY] = bas.rightX[leftY < rightY]
        bas.iter += 1
    print(bas.best)
