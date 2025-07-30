# RNN.py
import torch.nn as nn

class KeystrokeRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=2):
        super(KeystrokeRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = None
        out, _ = self.rnn(x, h_0)
        out = self.fc(out[:, -1, :])
        return out
