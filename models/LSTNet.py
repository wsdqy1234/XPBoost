import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTNet(nn.Module):

    def __init__(self, args, featureNum, outputNum):

        super(LSTNet, self).__init__()
        self.window = args.window
        self.featureNum = featureNum
        self.outputNum = outputNum
        self.hidR = args.hidR
        self.hidC = args.hidC
        self.hidS = args.hidSkip
        self.Ck = args.kernelCNN
        self.skip = args.skip
        self.pt = int((self.window - self.Ck) / self.skip)
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.featureNum))
        self.GRU1 = nn.GRU(self.hidC, self.hidR, num_layers=args.layerRNN, bidirectional=args.bidirection)
        self.dropout = nn.Dropout(args.dropout)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.FC = nn.Linear(self.hidR + self.skip * self.hidS, self.featureNum)
        else:
            self.FC = nn.Linear(self.hidR, self.featureNum)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.FC0 = nn.Linear(self.featureNum, args.dimFC)
        self.FC1 = nn.Linear(args.dimFC, args.dimFC)
        self.FC2 = nn.Linear(args.dimFC, self.outputNum)
        self.output = None

    def forward(self, x):

        batchSize = x.size(0)

        # CNN
        c = x.view(-1, 1, self.window, self.featureNum)  # (batch,1,p,m)
        c = F.relu(self.conv1(c))  # (batch,c_out,p-kernel_size,1)
        c = self.dropout(c)
        c = torch.squeeze(c, 3)  # (batch,c_out,p-kernel_size)

        # RNN 
        r = c.permute(2, 0, 1).contiguous()  # (time,batch,c_out)
        _, r = self.GRU1(r)  # (1, batch, c_out)
        r = self.dropout(torch.squeeze(r, 0))  # (batch, c_out)

        # skip-rnn

        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous()  # 截取可以周期分解（skip）的部分
            s = s.view(batchSize, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()  # pt, batch, len per pt, hid
            s = s.view(self.pt, batchSize * self.skip, self.hidC)
            _, s = self.GRUskip(s)  # (1, batch*skip, hidS)
            s = s.view(batchSize, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.FC(r)

        # highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.featureNum)
            res = res + z

        output = self.FC0(res)
        output = torch.tanh(output)
        output = self.dropout(output)
        output = self.FC1(output)
        output = torch.tanh(output)
        output = self.FC2(output)

        if (self.output):
            output = self.output(output)
        return output
