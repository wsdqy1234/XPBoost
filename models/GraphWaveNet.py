import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

from torch.nn.modules.activation import ReLU


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for A in support:
            x1 = self.nconv(x, A)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, A)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GraphWaveNet(nn.Module):
    def __init__(self, args, featureNum, outputNum, supports=None, aptinit=None):
        # def __init__(self, device, num_nodes, supports=None, dimI=2,dimO=12,chanRes=32,chanDil=32,chanSkip=256,chanEnd=512,kernelSize=2,stackSize=4,layerSize=2):
        super(GraphWaveNet, self).__init__()
        self.device = torch.device('cuda:{}'.format(args.device))
        self.dropout = nn.Dropout(args.dropout)
        self.stackSize = args.stackSize
        self.layerSize = args.layerSize
        self.kernelSize = args.kernelSize
        self.dimO = args.dimO
        self.chanRes = args.chanRes
        self.chanDil = args.chanDil
        self.chanSkip = args.chanSkip
        self.chanEnd = args.chanEnd
        self.gcn_bool = args.gcn_bool
        self.addaptadj = args.addaptadj

        self.supports = supports
        self.aptinit = aptinit

        self.numNodes = featureNum  # args.window: temporal nodes

        self.filter_convs = nn.ModuleList()
        if self.gcn_bool:
            self.gconv = nn.ModuleList()
        else:
            self.residual_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=1, out_channels=self.chanRes, kernel_size=(1, 1))
        self.supports_len = 0 if supports is None else len(supports)
        receptive_field = 1

        if self.gcn_bool and self.addaptadj:
            if supports is None:
                if self.aptinit is None:

                    self.supports = []
                    self.nodevec1 = nn.Parameter(torch.randn(self.numNodes, 10).to(self.device), requires_grad=True).to(
                        self.device)
                    self.nodevec2 = nn.Parameter(torch.randn(10, self.numNodes).to(self.device), requires_grad=True).to(
                        self.device)
                    self.supports_len += 1
                else:

                    self.supports = []
                    m, p, n = torch.svd(self.aptinit)
                    initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                    initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                    self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
                    self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)
                    self.supports_len += 1

        for b in range(self.stackSize):
            additional_scope = self.kernelSize - 1
            new_dilation = 1
            for i in range(self.layerSize):
                # dilated convolutions
                self.filter_convs.append(
                    nn.Conv2d(in_channels=self.chanRes, out_channels=self.chanDil, kernel_size=(1, self.kernelSize),
                              dilation=new_dilation))
                self.gate_convs.append(
                    nn.Conv1d(in_channels=self.chanRes, out_channels=self.chanDil, kernel_size=(1, self.kernelSize),
                              dilation=new_dilation))

                # 1x1 convolution for res/skip connection
                if self.gcn_bool:
                    self.gconv.append(GCN(self.chanDil, self.chanRes, args.dropout, support_len=self.supports_len))
                else:
                    self.residual_convs.append(
                        nn.Conv1d(in_channels=self.chanDil, out_channels=self.chanRes, kernel_size=(1, 1)))
                self.skip_convs.append(
                    nn.Conv1d(in_channels=self.chanDil, out_channels=self.chanSkip, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(self.chanRes))

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(in_channels=self.chanSkip, out_channels=self.chanEnd, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(in_channels=self.chanEnd, out_channels=self.dimO, kernel_size=(1, 1))
        self.receptive_field = receptive_field

        self.FC0 = nn.Linear(featureNum * self.dimO, args.dimFC)
        self.FC1 = nn.Linear(args.dimFC, args.dimFC)
        self.FC2 = nn.Linear(args.dimFC, outputNum)

    def forward(self, input):

        input = input.transpose(1, 2)
        input = input.unsqueeze(1)

        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)  # change the channels (feature num) to args.chanRes
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layerSize
        for i in range(self.stackSize * self.layerSize):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]

            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            residual = torch.nn.functional.pad(residual, (1, 0))
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:

                x = self.gconv[i](x, new_supports) if self.addaptadj else self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = self.end_conv_1(x)
        x = F.relu(x)
        x = self.end_conv_2(x)

        x = torch.flatten(x[:, :, :, -1], start_dim=1)
        x = self.FC0(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.FC1(x)
        x = torch.tanh(x)
        x = self.FC2(x)

        return x
