from typing import Literal

import torch
import torch.nn as nn

# class WeatherRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers):
#         super(WeatherRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)
#
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
#         output, _ = self.rnn(x, h0)
#         output = self.fc(output[:, -1, :])
#         return output


class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int, activation: Literal['tanh', 'sigmoid', 'relu'] = 'tanh'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

        self.rnn_layers = nn.ModuleList()

        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.rnn_layers.append(
                nn.ModuleDict({
                    "i2h": nn.Linear(in_size, hidden_size),
                    "h2h": nn.Linear(hidden_size, hidden_size)
                })
            )
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        h_t = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]

        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                i2h = self.rnn_layers[layer]['i2h']
                h2h = self.rnn_layers[layer]['h2h']

                h_t[layer] = self.activation(i2h(x_t) + h2h(h_t[layer]))
                x_t = h_t[layer]

        output = self.h2o(h_t[-1])
        return output
