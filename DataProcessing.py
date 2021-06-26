import pandas as pd

dataName = 'pems08speed'
root = '/home/wangh/TimeSeries/data/'
data = pd.read_csv(root + '{0}.txt'.format(dataName), header=None, index_col=None)
dataLen = len(data)
data.iloc[:int(dataLen * 0.7), :].to_csv(root + '{}Train.txt'.format(dataName), header=None, index=None)
data.iloc[int(dataLen * 0.7): int(dataLen * 0.85), :].to_csv(root + '{}Val.txt'.format(dataName), header=None,
                                                             index=None)
data.iloc[int(dataLen * 0.85):, :].to_csv(root + '{0}Test.txt'.format(dataName), header=None, index=None)

print(data)
