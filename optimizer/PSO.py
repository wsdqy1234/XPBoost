import multiprocessing
import numpy as np
from sklearn.cross_decomposition import PLSRegression
import os
import os.path as osp
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from multiprocessing.dummy import Pool, Manager


class PSO():
    """
    This is the original implementation of the Particle swarm optimization algorithm
    """

    def __init__(self, initpoint, shape_w1, args, trainX, trainY, validX, validY, testX, testY, plsLatent, lb=None,
                 ub=None):

        # multiprocessing.set_start_method('forkserver')
        self.logDir = osp.join('./works', args.dataName, args.modelName + args.taskName + str(args.horizon),
                               str(args.seed))
        if osp.exists(self.logDir) != True:
            os.makedirs(self.logDir)
        self.logTrain = osp.join(self.logDir, 'pso_gbest_train.txt')
        self.logValid = osp.join(self.logDir, 'pso_gbest_val.txt')
        self.logTest = osp.join(self.logDir, 'pso_gbest_test.txt')

        for item in [self.logTrain, self.logValid, self.logTest]:
            if osp.exists(item):
                os.remove(item)

        self.shapeW1 = shape_w1
        self.shape1, self.shape2 = shape_w1

        self.data = {'trainX': trainX, 'trainY': trainY, 'validX': validX, 'validY': validY, 'testX': testX,
                     'testY': testY}

        self.cp, self.cg, self.w = args.pso_c1, args.pso_c2, args.pso_w  # parameters to control personal best, global best respectively
        self.pop = args.pso_pop  # number of particles
        self.max_iter = args.pso_iter  # max iter
        self.dim = (self.shapeW1[0] + 1) * self.shapeW1[1]
        self.iter = 0
        self.has_constraints = not (lb is None and ub is None)
        self.lb = -np.ones((self.pop, self.dim)) if lb is None else np.array(lb)
        self.ub = np.ones((self.pop, self.dim)) if ub is None else np.array(ub)
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.plsLatent = plsLatent
        self.plsIter = args.plsIter
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        self.X[0, :] = initpoint
        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim))  # speed of particles
        self.Y = None
        self.cal_y()  # y = f(x) for all particles

        self.pbest = {'X': self.X.copy(), 'Y': self.Y.copy()}
        self.gbest = {'X': np.zeros((1, self.dim)), 'trainLoss': np.inf, 'trainR2': None, 'valLoss': None,
                      'valR2': None, 'testLoss': None, 'testR2': None}
        self.update_gbest()

    def update_V(self):
        r1 = np.random.rand(self.pop, self.dim)
        r2 = np.random.rand(self.pop, self.dim)
        self.V = self.w * self.V + self.cp * r1 * (self.pbest['X'] - self.X) + self.cg * r2 * (self.gbest['X'] - self.X)

    def update_X(self):
        self.X = self.X + self.V
        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        # calculate y for every x in X
        self.pls_fit()

    def update_pbest(self):

        for i in range(len(self.Y)):
            if self.pbest['Y'][i] > self.Y[i]:
                self.pbest['X'][i, :] = self.X[i, :].copy()
                self.pbest['Y'][i] = self.Y[i].copy()

    def update_gbest(self):

        if self.gbest['trainLoss'] > self.Y.min():
            self.gbest['X'] = self.X[self.Y.argmin(), :].copy()
            self.gbest['trainLoss'] = self.Y.min()

            with open(self.logTrain, 'a') as f:
                temp = np.insert(self.gbest['X'], 0, self.gbest['trainLoss'])
                temp = np.insert(temp, 0, self.iter).reshape(1, -1)
                np.savetxt(f, temp, delimiter=',')
                f.write('\n')
            self.gbest['valLoss'], self.gbest['valR2'] = self.pls_val(self.data['validX'], self.data['validY'])
            with open(self.logValid, 'a') as f:
                f.write('{0},{1},{2}\n'.format(self.iter, self.gbest['valLoss'], self.gbest['valR2']))
            self.gbest['testLoss'], self.gbest['testR2'] = self.pls_val(self.data['testX'], self.data['testY'])
            with open(self.logTest, 'a') as f:
                f.write('{0},{1},{2}\n'.format(self.iter, self.gbest['testLoss'], self.gbest['testR2']))

            print(
                'gbestY_train={0}, gbestY_test={1}, r2_test={2}'.format(self.gbest['trainLoss'], self.gbest['testLoss'],
                                                                        self.gbest['testR2']))

    def run(self):

        while self.iter <= self.max_iter:
            self.update_V()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()

            # self.gbest['trainLoss']_hist.append(self.gbest['trainLoss'])
            self.iter += 1
            print(self.iter)

    def pls_fit_single(self):

        vector = self.X.copy()

        X = []
        trainLoss = []
        trainR2 = []
        weight1Len = self.shapeW1[0] * self.shapeW1[1]

        for item in vector:
            weight1 = item[:weight1Len].reshape((self.shapeW1[0], self.shapeW1[1]))
            bias1 = item[weight1Len:].reshape((1, self.shapeW1[1]))
            a = np.tanh(np.matmul(self.data['trainX'], weight1) + bias1)

            pls = PLSRegression(n_components=self.plsLatent, max_iter=self.plsIter)
            # pls = LinearRegression()
            pls.fit(a, self.data['trainY'])
            predY = pls.predict(self.data['trainX'])
            train_loss = np.sqrt(mean_squared_error(self.data['trainY'], predY))
            r2 = r2_score(self.data['trainY'], predY)

            X.append(item)
            trainLoss.append(train_loss)
            trainR2.append(r2)

        self.X = np.array(X)
        self.Y = np.array(trainLoss)

    def pls_fit(self):

        vector = self.X.copy()
        manager = Manager()
        numProcess = 18
        subLength = int(len(vector) // numProcess)
        xDict = manager.dict()  # loss for the total swarm
        trainLossDict = manager.dict()  # loss for the total swarm
        trainR2Dict = manager.dict()

        def meta(plsLatent, plsIter, subVector, shapeW1, data, xDict, trainLossDict, trainR2Dict, idx):
            print('start')
            X = []
            trainLoss = []
            trainR2 = []
            weight1Len = shapeW1[0] * shapeW1[1]

            for item in subVector:
                weight1 = item[:weight1Len].reshape((shapeW1[0], shapeW1[1]))
                bias1 = item[weight1Len:].reshape((1, shapeW1[1]))
                a = np.tanh(np.matmul(data['trainX'], weight1) + bias1)

                pls = PLSRegression(n_components=plsLatent, max_iter=plsIter)
                # pls = LinearRegression()
                pls.fit(a, data['trainY'])
                predY = pls.predict(data['trainX'])
                train_loss = np.sqrt(mean_squared_error(data['trainY'], predY))
                r2 = r2_score(data['trainY'], predY)

                X.append(item)
                trainLoss.append(train_loss)
                trainR2.append(r2)

            xDict[idx] = X
            trainLossDict[idx] = trainLoss
            trainR2Dict[idx] = trainR2
            print('end')

        pool = Pool(processes=50)
        for idx in range(numProcess):
            subVector = vector[idx * subLength: (idx + 1) * subLength, :]
            pool.apply_async(meta, (
            self.plsLatent, self.plsIter, subVector, self.shapeW1, self.data, xDict, trainLossDict, trainR2Dict, idx))
        pool.close()
        pool.join()

        # processList = []
        # for idx in range(numProcess):
        #     subVector = vector[idx*subLength: (idx+1)*subLength, :]
        #     p = multiprocessing.Process(target=meta, args=(self.plsLatent, self.plsIter, subVector, self.data['trainY'], xDict, trainLossDict, trainR2Dict, idx))
        #     processList.append(p)
        # [x.start() for x in processList]
        # [x.join() for x in processList]

        for i in range(numProcess):
            if i == 0:
                X = xDict[i]
                Y = trainLossDict[i]
            else:
                X += xDict[i]
                Y += trainLossDict[i]

        self.X = np.array(X)
        self.Y = np.array(Y)

    def forward(self, input):

        weightVector = self.gbest['X']
        weight1 = weightVector[:self.shapeW1[0] * self.shapeW1[1]].reshape(self.shapeW1)
        bias1 = weightVector[self.shapeW1[0] * self.shapeW1[1]:].reshape(1, self.shapeW1[1])
        a = np.matmul(input, weight1) + bias1
        a = np.tanh(a)
        return a

    def pls_val(self, X, Y):

        trainInput = self.forward(self.data['trainX'])
        # trainInput = self.forward()
        pls = PLSRegression(n_components=self.plsLatent, max_iter=self.plsIter)
        # pls = LinearRegression()
        pls.fit(trainInput, self.data['trainY'])
        # pls.fit(trainInput, Y)

        Input = self.forward(X)
        Output = pls.predict(Input)
        rmse = np.sqrt(mean_squared_error(Y, Output))
        r2 = r2_score(Y, Output)

        return rmse, r2
