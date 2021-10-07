import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SpectralCRNN(nn.Module):
    def __init__(self):
        super(SpectralCRNN, self).__init__()
        self.conv = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 16, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d((2, 4)),
            # Conv Layer 2
            nn.Conv2d(16, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((3, 5)),
            # Conv Layer 3
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((3, 5)))
        self.rnn = nn.GRU(320, 200, batch_first=True)
        self.fc = nn.Linear(200, 11)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1, out.size(3))
        out = out.transpose(1, 2)
        out, _ = self.rnn(out, self.hidden)
        out = out[:, -1, :]
        out = self.fc(out)
        return out  #F.log_softmax(out)

    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the PitchContourAssessor module
        Args:
                mini_batch_size:    number of data samples in the mini-batch
        """
        self.hidden = Variable(torch.zeros(1, mini_batch_size, 200))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()


class SpectralCRNN_Reg(nn.Module):
    def __init__(self):
        super(SpectralCRNN_Reg, self).__init__()
        self.conv = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 16, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d((2, 4)),
            # Conv Layer 2
            nn.Conv2d(16, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((3, 5)),
            # Conv Layer 3
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((3, 5)))
        self.rnn = nn.GRU(320, 200, batch_first=True)
        self.fc = nn.Linear(200, 1)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1, out.size(3))
        out = out.transpose(1, 2)
        out, _ = self.rnn(out, self.hidden)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the PitchContourAssessor module
        Args:
                mini_batch_size:    number of data samples in the mini-batch
        """
        self.hidden = Variable(torch.zeros(1, mini_batch_size, 200))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()


class SpectralCRNN_Reg_Dropout(nn.Module):
    def __init__(self):
        super(SpectralCRNN_Reg_Dropout, self).__init__()
        self.conv = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((2, 4)),
            # Conv Layer 2
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((3, 5)),
            # Conv Layer 3
            nn.Conv2d(64, 128, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d((3, 5)))
        self.rnn = nn.GRU(640, 200, batch_first=True)
        self.fc = nn.Linear(200, 1)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1, out.size(3))
        out = out.transpose(1, 2)
        out, _ = self.rnn(out, self.hidden)
        out = out[:, -1, :]
        out = self.fc(out)
        return out  #F.relu(out)

    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the PitchContourAssessor module
        Args:
                mini_batch_size:    number of data samples in the mini-batch
        """
        self.hidden = Variable(torch.zeros(1, mini_batch_size, 200))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()


class SpectralCRNN_Reg_Dropout_mel_ph(nn.Module):
    def __init__(self):
        super(SpectralCRNN_Reg_Dropout_mel_ph, self).__init__()
        self.conv = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((2, 4)),
            # Conv Layer 2
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((3, 5)),
            # Conv Layer 3
            nn.Conv2d(64, 128, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d((3, 5)))
        self.rnn = nn.GRU(640, 200, batch_first=True)
        #self.fc_mel = nn.Linear(200,120)
        self.fc = nn.Linear(320, 1)

    def forward(self, x, ph): # 输入data和ph
        out = self.conv(x)
        out = out.view(out.size(0), -1, out.size(3))
        out = out.transpose(1, 2)
        out, _ = self.rnn(out, self.hidden)
        out = out[:, -1, :]
        #mel = self.fc_mel(out)
        feature = torch.cat((out, ph), 1)
        out = self.fc(feature)
        return out, feature

    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the PitchContourAssessor module
        Args:
                mini_batch_size:    number of data samples in the mini-batch
        """
        self.hidden = Variable(torch.zeros(1, mini_batch_size, 200))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()


class SpectralCRNN_Reg_Dropout_CQT_ph(nn.Module):
    def __init__(self):
        super(SpectralCRNN_Reg_Dropout_CQT_ph, self).__init__()
        self.conv1 = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((2, 4)),
            # Conv Layer 2
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((3, 5)),
            # Conv Layer 3
            nn.Conv2d(64, 128, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d((3, 5)))
        self.conv2 = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((2, 4)),
            # Conv Layer 2
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((3, 5)),
            # Conv Layer 3
            nn.Conv2d(64, 128, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d((3, 5)))
        self.rnn = nn.GRU(640, 200, batch_first=True)
        self.fc_ps = nn.Linear(320, 1)
        self.fc_all = nn.Linear(200, 1)


    def forward(self, x, ph):
        out1 = self.conv1(x)
        out1 = out1.view(out1.size(0), -1, out1.size(3))
        out1 = out1.transpose(1, 2)
        out1, _ = self.rnn(out1, self.hidden)
        out1 = out1[:, -1, :]

        out2 = self.conv2(x)
        out2 = out2.view(out2.size(0), -1, out2.size(3))
        out2 = out2.transpose(1, 2)
        out2, _ = self.rnn(out2, self.hidden)
        out2 = out2[:, -1, :]

        feature_ph = torch.cat((out1, ph), 1)
        out_ps = self.fc_ps(feature_ph)
        feature_all = out2
        out_all = self.fc_all(feature_all)
        return [out_all,out_ps], feature_ph

    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the PitchContourAssessor module
        Args:
                mini_batch_size:    number of data samples in the mini-batch
        """
        self.hidden = Variable(torch.zeros(1, mini_batch_size, 200))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()




class SpectralCRNN_Reg_big(nn.Module):
    def __init__(self):
        super(SpectralCRNN_Reg_big, self).__init__()
        self.conv = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 48, kernel_size=(3, 7), padding=(1, 3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(48),
            nn.ELU(),
            nn.MaxPool2d((2, 4)),
            # Conv Layer 2
            nn.Conv2d(48, 64, kernel_size=(3, 7), padding=(1, 3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((3, 5)),
            # Conv Layer 3
            nn.Conv2d(64, 128, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d((3, 5)))
        self.rnn = nn.GRU(640, 200, batch_first=True)
        self.fc = nn.Linear(200, 1)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1, out.size(3))
        out = out.transpose(1, 2)
        out, _ = self.rnn(out, self.hidden)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the PitchContourAssessor module
        Args:
                mini_batch_size:    number of data samples in the mini-batch
        """
        self.hidden = Variable(torch.zeros(1, mini_batch_size, 200))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()


class SpectralCRNN_Reg_128_mels(nn.Module):
    def __init__(self):
        super(SpectralCRNN_Reg_128_mels, self).__init__()
        self.conv = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((2, 4)),
            # Conv Layer 2
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((3, 5)),
            # Conv Layer 3
            nn.Conv2d(64, 128, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d((3, 5)))
        self.rnn = nn.GRU(896, 200, batch_first=True)
        self.fc = nn.Linear(200, 1)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1, out.size(3))
        out = out.transpose(1, 2)
        out, _ = self.rnn(out, self.hidden)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the PitchContourAssessor module
        Args:
                mini_batch_size:    number of data samples in the mini-batch
        """
        self.hidden = Variable(torch.zeros(1, mini_batch_size, 200))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()


class SpectralCRNN_Reg_PRELU(nn.Module):
    def __init__(self):
        super(SpectralCRNN_Reg_PRELU, self).__init__()
        self.conv = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d((2, 4)),
            # Conv Layer 2
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d((3, 5)),
            # Conv Layer 3
            nn.Conv2d(64, 128, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d((3, 5)))
        self.rnn = nn.GRU(640, 200, batch_first=True)
        self.fc = nn.Linear(200, 1)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1, out.size(3))
        out = out.transpose(1, 2)
        out, _ = self.rnn(out, self.hidden)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the PitchContourAssessor module
        Args:
                mini_batch_size:    number of data samples in the mini-batch
        """
        self.hidden = Variable(torch.zeros(1, mini_batch_size, 200))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()


class SpectralCRNN_Reg_PRELU_diff_filter(nn.Module):
    def __init__(self):
        super(SpectralCRNN_Reg_PRELU_diff_filter, self).__init__()
        self.conv = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 32, kernel_size=(21, 7), padding=(10, 3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d((2, 4)),
            # Conv Layer 2
            nn.Conv2d(32, 64, kernel_size=(11, 7), padding=(5, 3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d((3, 5)),
            # Conv Layer 3
            nn.Conv2d(64, 128, kernel_size=(7, 7), padding=(3, 3)),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d((3, 5)))
        self.rnn = nn.GRU(640, 200, batch_first=True)
        self.fc = nn.Linear(200, 1)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1, out.size(3))
        out = out.transpose(1, 2)
        out, _ = self.rnn(out, self.hidden)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the PitchContourAssessor module
        Args:
                mini_batch_size:    number of data samples in the mini-batch
        """
        self.hidden = Variable(torch.zeros(1, mini_batch_size, 200))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()
