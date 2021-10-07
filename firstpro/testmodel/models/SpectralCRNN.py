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

    def forward(self, x):  # x: mel spectrogram,2D
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


class SpectralCRNN_Reg_Dropout_tsne(nn.Module):
    def __init__(self):
        super(SpectralCRNN_Reg_Dropout_tsne, self).__init__()
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
        feature = out
        out = self.fc(out)
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
