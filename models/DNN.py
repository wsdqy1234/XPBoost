import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, args, featureNum, outputNum):

        super(DNN, self).__init__()
        layers = []
        for i in range(len(args.nodes)):
            if i == 0:
                layers += [nn.Linear(featureNum, args.nodes[0])]
            else:
                layers += [nn.Linear(args.nodes[i - 1], args.nodes[i])]

        self.network = nn.Sequential(*layers)
        self.dropout = nn.Dropout(args.dropout)
        self.FC0 = nn.Linear(args.nodes[i], args.dimFC)
        self.FC1 = nn.Linear(args.dimFC, args.dimFC)
        self.FC2 = nn.Linear(args.dimFC, outputNum)

    def forward(self, input):

        output = input.squeeze()
        output = self.network(output)
        output = self.FC0(output)
        output = torch.tanh(output)
        output = self.dropout(output)
        output = self.FC1(output)
        output = torch.tanh(output)
        output = self.FC2(output)

        return output
