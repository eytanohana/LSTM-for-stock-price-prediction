import numpy as np
import pandas as pd
import math
import sklearn
from sklearn.preprocessing import StandardScaler
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.ticker as mticker
from mpl_finance import candlestick_ohlc
import yfinance as yf

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.nn.functional as F

import matplotlib.pyplot as plt


def create_sequences(input_data, seq_len, close_index):
    seq = []
    L = len(input_data)
    for i in range(L-seq_len):
        train_seq = input_data[i:i+seq_len]
        train_label = input_data[i+seq_len:i+seq_len+1][0,close_index]
        seq.append((train_seq ,train_label))
    return seq


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.h0 = torch.zeros(1, 1, hidden_size, device=device)
        self.c0 = torch.zeros(1, 1, hidden_size, device=device)
        
    def forward(self, x):        
        out_lstm, (self.h0, self.c0) = self.lstm(x.view(len(x), 1, -1), (self.h0, self.c0))
        out = self.fc(out_lstm.view(len(x), -1))
        return out[-1][0]


class LSTM_linear_before_after(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_linear_before_after, self).__init__()

        self.fc2 = nn.Linear(input_size, 32)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(32, hidden_size, bidirectional=False)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h0 = torch.zeros(1, 1, hidden_size, device=self.device)
        self.c0 = torch.zeros(1, 1, hidden_size, device=self.device)

    def forward(self, x):     
        y = self.fc2(x.view(len(x), -1))
        y_relu = F.sigmoid(y)
        out_lstm, (self.h0, self.c0) = self.lstm(y_relu.view(len(x), 1, -1), (self.h0, self.c0))
        out = self.fc1(out_lstm.view(len(x), -1))
        return out[-1][0]

    
class LSTM_2_layer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.h0 = torch.zeros(num_layers, 1, hidden_size, device=device)
        self.c0 = torch.zeros(num_layers, 1, hidden_size, device=device)

    def forward(self, x):       
        out_lstm, (self.h0, self.c0) = self.lstm(x.view(len(x),1,-1), (self.h0, self.c0))
        out = self.fc(out_lstm.view(len(out_lstm), -1))
        return out[-1][0]
