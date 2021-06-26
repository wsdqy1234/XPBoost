import torch
import numpy as np


class DilatedCausalConv1d(torch.nn.Module):
    """Dilated Causal Convolution for WaveNet"""

    def __init__(self, channels, dilation=1):
        super(DilatedCausalConv1d, self).__init__()

        self.conv = torch.nn.Conv1d(channels, channels, kernel_size=2, stride=1, padding=0, bias=False,
                                    # Fixed for WaveNet
                                    dilation=dilation)

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)

        return output


class CausalConv1d(torch.nn.Module):
    """Causal Convolution for WaveNet"""

    def __init__(self, chanI, out_channels):
        super(CausalConv1d, self).__init__()

        # padding=1 for same size(length) between input and output for causal convolution
        self.conv = torch.nn.Conv1d(chanI, out_channels,
                                    kernel_size=2, stride=1, padding=1,
                                    bias=False)  # Fixed for WaveNet but not sure

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)  # (4, 100, 169)

        # remove last value for causal convolution
        return output[:, :, :-1]  # (4, 100, 168)


class ResidualBlock(torch.nn.Module):
    def __init__(self, chanRes, chanSkip, dilation, skipSize):
        """
        Residual block
        :param chanRes: number of residual channel for input, output
        :param chanSkip: number of skip channel for output
        :param dilation:
        """
        super(ResidualBlock, self).__init__()

        self.skipSize = skipSize

        self.dilated = DilatedCausalConv1d(chanRes, dilation=dilation)
        self.conv_res = torch.nn.Conv1d(chanRes, chanRes, 1)
        self.conv_skip = torch.nn.Conv1d(chanRes, chanSkip, 1)

        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        :param x:
        :param skipSize: The last output size for loss and prediction
        :return:
        """
        output = self.dilated(x)  # (4, 100, 168-1-2-4)

        # PixelCNN gate
        gated_tanh = self.gate_tanh(output)
        gated_sigmoid = self.gate_sigmoid(output)
        gated = gated_tanh * gated_sigmoid

        # Residual network
        output = self.conv_res(gated)  # (4, 100, 168-1-2-4)
        input_cut = x[:, :, -output.size(
            2):]  # introduce the information of x in the correspounding steps. (early information is dropped)
        output += input_cut

        # Skip connection
        skip = self.conv_skip(gated)
        skip = skip[:, :, -self.skipSize:]  # (4, 100, 168-1-2-4)

        return output, skip


class ResidualStack(torch.nn.Module):
    def __init__(self, layerSize, stackSize, chanRes, chanSkip, skipSize, device):
        """
        Stack residual blocks by layer and stack size
        :param layerSize: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stackSize: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param chanRes: number of residual channel for input, output
        :param chanSkip: number of skip channel for output
        :return:
        """
        super(ResidualStack, self).__init__()

        self.device = device
        self.layerSize = layerSize
        self.stackSize = stackSize
        self.skipSize = skipSize
        self.resBlocks = self.stack_res_block(chanRes, chanSkip)

    def build_dilations(self):
        dilations = []

        for s in range(0, self.stackSize):
            # 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
            for l in range(0, self.layerSize):
                dilations.append(2 ** l)

        return dilations

    def stack_res_block(self, chanRes, chanSkip):

        resBlocks = []
        dilations = self.build_dilations()

        for dilation in dilations:
            block = ResidualBlock(chanRes, chanSkip, dilation, self.skipSize).to(self.device)
            resBlocks.append(block)

        return resBlocks

    def forward(self, x):

        output = x
        skip_connections = []

        for res_block in self.resBlocks:
            # output is the next input
            output, skip = res_block(output)
            skip_connections.append(skip)

        return torch.stack(skip_connections)


class DenseNet(torch.nn.Module):
    def __init__(self, channels):
        """
        The last network of WaveNet
        :param channels: number of channels for input and output
        :return:
        """
        super(DenseNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(channels, channels, 1)
        self.conv2 = torch.nn.Conv1d(channels, channels, 1)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        output = self.relu(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)

        output = self.softmax(output)

        return output


class WaveNet(torch.nn.Module):
    def __init__(self, args, featureNum, outputNum):
        """
        Stack residual blocks by layer and stack size
        :param layerSize: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stackSize: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param chanI: number of channels for input data. skip channel is same as input channel
        :param chanRes: number of residual channel for input, output
        :return:
        """
        super(WaveNet, self).__init__()

        self.layerSize = args.layerSize
        self.stackSize = args.stackSize
        self.chanI = featureNum
        self.chanRes = args.chanRes
        self.chanSkip = args.chanSkip
        self.device = torch.device('cuda:{}'.format(args.device))

        self.receptive_fields = np.sum([2 ** i for i in range(0, self.layerSize)] * self.stackSize)
        self.skipSize = args.window - self.receptive_fields

        self.causal = CausalConv1d(self.chanI, self.chanRes)
        self.res_stack = ResidualStack(self.layerSize, self.stackSize, self.chanRes, self.chanSkip, self.skipSize,
                                       self.device)
        self.densenet = DenseNet(self.chanSkip)
        self.FC0 = torch.nn.Linear(self.skipSize * self.chanSkip, args.dimFC)
        self.FC1 = torch.nn.Linear(args.dimFC, args.dimFC)
        self.FC2 = torch.nn.Linear(args.dimFC, outputNum)
        self.dropout = torch.nn.Dropout(args.dropout)

    def forward(self, x):
        """
        The size of timestep(3rd dimention) has to be bigger than receptive fields
        :param x: Tensor[batch, timestep, channels]
        :return: Tensor[batch, timestep, channels]
        """
        output = x.transpose(1, 2)  # (4, 862, 168)
        output = self.causal(output)  # (4, 100, 168)
        skip_connections = self.res_stack(output)  # (layerSize, 4, 100, xx)
        output = torch.sum(skip_connections, dim=0)
        output = self.densenet(output)
        output.transpose(1, 2).contiguous()
        output = output.reshape((-1, self.skipSize * self.chanSkip))

        output = self.FC0(output)
        output = torch.tanh(output)
        output = self.dropout(output)

        output = self.FC1(output)
        output = torch.tanh(output)

        output = self.FC2(output)

        return output
