import torch
import torch.nn as nn


class WeatherRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(WeatherRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        output, _ = self.rnn(x, h0)
        output = self.fc(output[:, -1, :])
        return output

