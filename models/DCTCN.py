import torch
import numpy as np
import torch.nn as nn


class DilatedCausalConv1d(torch.nn.Module):
    """Dilated Causal Convolution for WaveNet"""

    def __init__(self, inchannel, outchannel, dilation=1):
        super(DilatedCausalConv1d, self).__init__()

        self.conv = torch.nn.Conv1d(inchannel, outchannel,
                                    kernel_size=2, stride=1,  # Fixed for WaveNet
                                    dilation=dilation,
                                    padding=0,  # Fixed for WaveNet dilation
                                    bias=False)  # Fixed for WaveNet but not sure

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)

        return output


class CausalConv1d(torch.nn.Module):
    """Causal Convolution for WaveNet"""

    def __init__(self, in_channels, out_channels):
        super(CausalConv1d, self).__init__()

        # padding=1 for same size(length) between input and output for causal convolution
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=2, stride=1, padding=1,
                                    bias=False)  # Fixed for WaveNet but not sure

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)

        # remove last value for causal convolution
        return output[:, :, :-1]


class DCTCN(nn.Module):
    def __init__(self, args, featureNum, outputNum):
        super(DCTCN, self).__init__()

        self.tcn1 = DilatedCausalConv1d(featureNum, args.dimC, dilation=1)
        self.tcn2 = DilatedCausalConv1d(args.dimC, args.dimC, dilation=2)
        self.tcn3 = DilatedCausalConv1d(args.dimC, args.dimC, dilation=4)
        self.tcn4 = DilatedCausalConv1d(args.dimC, args.dimC, dilation=8)
        self.tcn5 = DilatedCausalConv1d(args.dimC, int(args.dimC / 2), dilation=16)
        self.dropout = nn.Dropout(args.dropout)
        self.FC1 = torch.nn.Linear(int(args.dimC / 2) + featureNum, args.dimFC)
        self.FC2 = torch.nn.Linear(args.dimFC, outputNum)

    def forward(self, input):
        input = torch.transpose(input, 1, 2)
        res = input.clone()[:, :, -1]
        output = torch.relu(self.tcn1(input))
        output = torch.relu(self.tcn2(output))
        output = torch.relu(self.tcn3(output))
        output = torch.relu(self.tcn4(output))
        output = torch.relu(self.tcn5(output))

        output = output[:, :, -1]
        output = torch.cat([output, res], axis=1)
        output = torch.relu(self.FC1(output))
        output = self.dropout(output)
        output = self.FC2(output)

        return output
