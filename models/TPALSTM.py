import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TPALSTM(nn.Module):
    def __init__(self, args, featureNum, outputNum):
        super(TPALSTM, self).__init__()
        self.device = torch.device('cuda:{}'.format(args.device))
        self.featureNum = featureNum
        self.outputNum = outputNum
        self.window = args.window;  # window, read about it after understanding the flow of the code...What is window size? --- temporal window size (default 24 hours * 7)
        self.hw = args.highway_window

        self.hidC = args.hidC;
        self.hidLSTM = args.hidLSTM
        self.hidR = args.hidR;
        self.hidS = args.hidSkip;
        self.Ck = args.kernelCNN;  # the kernel size of the CNN layers
        self.skip = args.skip;
        self.layerRNN = args.layerRNN

        self.pt = (self.window - self.Ck) // self.skip  # period number
        self.lstm = nn.LSTM(input_size=featureNum, hidden_size=self.hidLSTM,
                            num_layers=self.layerRNN,
                            bidirectional=args.bidirection);
        self.compute_convolution = nn.Conv2d(1, self.hidC, kernel_size=(
            self.Ck, self.hidLSTM))  # hidC are the num of filters, default value of Ck is one
        self.attention_matrix = nn.Parameter(
            torch.ones(args.batchSize, self.hidC, self.hidLSTM, requires_grad=True, device=self.device))
        self.context_vector_matrix = nn.Parameter(
            torch.ones(args.batchSize, self.hidLSTM, self.hidC, requires_grad=True, device=self.device))
        self.final_state_matrix = nn.Parameter(
            torch.ones(args.batchSize, self.hidLSTM, self.hidLSTM, requires_grad=True, device=self.device))
        self.final_matrix = nn.Parameter(
            torch.ones(args.batchSize, self.featureNum, self.hidLSTM, requires_grad=True, device=self.device))
        torch.nn.init.xavier_uniform_(self.attention_matrix)
        torch.nn.init.xavier_uniform_(self.context_vector_matrix)
        torch.nn.init.xavier_uniform_(self.final_state_matrix)
        torch.nn.init.xavier_uniform_(self.final_matrix)
        # self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.featureNum));  # kernel size is size for the filters
        # self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p=args.dropout);
        # if (self.skip > 0):
        #     self.GRUskip = nn.GRU(self.hidC, self.hidS);
        #     self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.featureNum);
        # else:
        #     self.linear1 = nn.Linear(self.hidR, self.featureNum);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.FC0 = nn.Linear(self.featureNum, args.dimFC)
        self.FC1 = nn.Linear(args.dimFC, args.dimFC)
        self.FC2 = nn.Linear(args.dimFC, self.outputNum)

    def forward(self, input):
        batch_size = input.size(0);

        """
           Step 1. First step is to feed this information to LSTM and find out the hidden states 
        """
        input_to_lstm = input.permute(1, 0, 2).contiguous()
        lstm_hidden_states, (h_all, c_all) = self.lstm(input_to_lstm)  # lstm: (seq, batch, hidLSTM*direction)
        hn = h_all[-1].view(1, h_all.size(1), h_all.size(2))  # (1, batch, hidLSTM)

        """
            Step 2. Apply convolution on these hidden states. As in the paper TPA-LSTM, these filters are applied on the rows of the hidden state
        """
        output_realigned = lstm_hidden_states.permute(1, 0, 2).contiguous()  # (batch, seq, hidLSTM*direction)
        hn = hn.permute(1, 0, 2).contiguous()  # (batch, 1, hidLSTM)
        # cn = cn.permute(1, 0, 2).contiguous()
        input_to_convolution_layer = output_realigned.view(-1, 1, self.window, self.hidLSTM);
        convolution_output = F.relu(
            self.compute_convolution(input_to_convolution_layer));  # (batch, hidC, window-kernelCNN, 1)
        convolution_output = self.dropout(convolution_output);

        """
            Step 3. Apply attention on this convolution_output
        """
        convolution_output = convolution_output.squeeze(3)

        """
                In the next 10 lines, padding is done to make all the batch sizes as the same so that they do not pose any problem while matrix multiplication
                padding is necessary to make all batches of equal size
        """
        self.hnFinal = torch.zeros(self.attention_matrix.size(0), 1, self.hidLSTM).to(self.device)
        final_convolution_output = torch.zeros(self.attention_matrix.size(0), self.hidC, self.window - self.Ck + 1).to(
            self.device)
        diff = 0
        if (hn.size(0) < self.attention_matrix.size(0)):
            self.hnFinal[:hn.size(0), :, :] = hn
            final_convolution_output[:convolution_output.size(0), :, :] = convolution_output
            diff = self.attention_matrix.size(0) - hn.size(0)
        else:
            self.hnFinal = hn
            final_convolution_output = convolution_output

        """
           self.hnFinal, final_convolution_output are the matrices to be used from here on
        """
        convolution_output_for_scoring = final_convolution_output.permute(0, 2, 1).contiguous()
        final_hn_realigned = self.hnFinal.permute(0, 2, 1).contiguous()
        convolution_output_for_scoring = convolution_output_for_scoring
        final_hn_realigned = final_hn_realigned
        mat1 = torch.bmm(convolution_output_for_scoring, self.attention_matrix)
        scoring_function = torch.bmm(mat1, final_hn_realigned)
        alpha = torch.sigmoid(scoring_function)
        context_vector = alpha * convolution_output_for_scoring
        context_vector = torch.sum(context_vector, dim=1)

        """
           Step 4. Compute the output based upon final_hn_realigned, context_vector
        """
        context_vector = context_vector.view(-1, self.hidC, 1)
        h_intermediate = torch.bmm(self.final_state_matrix, final_hn_realigned) + torch.bmm(self.context_vector_matrix,
                                                                                            context_vector)
        result = torch.bmm(self.final_matrix, h_intermediate)
        result = result.permute(0, 2, 1).contiguous()
        result = result.squeeze()

        """
           Remove from result the extra result points which were added as a result of padding 
        """
        final_result = result[:result.size(0) - diff]

        """
        Adding highway network to it
        """

        if (self.hw > 0):
            z = input[:, -self.hw:, :];
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1, self.featureNum)
            final_result = final_result + z
        output = self.FC0(final_result)
        output = torch.tanh(output)
        output = self.dropout(output)
        output = self.FC1(output)
        output = torch.tanh(output)
        output = self.FC2(output)
        return output
