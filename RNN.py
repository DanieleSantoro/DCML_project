import torch
import torch.nn as nn

class KeystrokeRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2):
        super(KeystrokeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # BatchNorm per il layer hidden output
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.rnn(x, (h0, c0))  # out: (batch_size, seq_len, hidden_size)
        out = out[:, -1, :]  # prendo l’output dell’ultimo step temporale

        # BatchNorm: deve essere (batch_size, features) ma batchnorm1d lavora su (batch_size, features)
        out = self.batch_norm(out)

        out = self.fc(out)
        return out
