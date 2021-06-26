import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):

    def __init__(self, args, featureNum, outputNum):
        super(GRU, self).__init__()

        self.window = args.window
        self.featureNum = featureNum
        self.outputNum = outputNum
        self.hidR = args.hidR
        self.GRU = nn.GRU(input_size=self.featureNum, hidden_size=self.hidR, num_layers=args.layerRNN,
                          bidirectional=args.bidirection);
        self.dropout = nn.Dropout(p=args.dropout)
        self.FC0 = nn.Linear(self.hidR, args.dimFC)
        self.FC1 = nn.Linear(args.dimFC, args.dimFC)
        self.FC2 = nn.Linear(args.dimFC, self.outputNum)

    def forward(self, x):
        # CNN
        c = x.view(-1, self.window, self.featureNum);  # (batch,p,m=1)
        r = c.permute(1, 0, 2).contiguous();  # (time,batch,c_out)
        _, r = self.GRU(r);  # (1, batch, c_out)
        r = torch.squeeze(r, 0)  # (batch, c_out)
        r = self.FC0(r)
        r = torch.tanh(r)
        r = self.dropout(r)
        r = self.FC1(r)
        r = torch.tanh(r)
        r = self.FC2(r)

        return r
