from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (mean_absolute_percentage_error,
                             root_mean_squared_error)
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from torch.optim.adam import Adam
from tqdm import tqdm

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


class LSTM(nn.Module):
    def __init__(self, num_classes: int = 1,
                 input_size: int = 1,
                 hidden_size: int = 16,
                 num_layers: int = 1,
                 seq_length: int = 7*24*12):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


def sliding_windows(data: np.ndarray, seq_length: int):
    x = []
    y = []

    for i in range(len(data)-seq_length-1-seq_length-7*24*12):
        _x = data[i:(i+seq_length)]
        x.append(_x)

        # y is the max of the respective next 7th day
        _y = np.max(data[i+seq_length+7*24*12: i +
                    seq_length+7*24*12+seq_length])
        y.append(_y)

    return np.array(x), np.array(y)


def load_data(base_path: Path, ip: str, seq_length: int = 7*24*12):
    df = pd.read_csv(base_path.joinpath(
        f"{ip}.csv"), parse_dates=['timestamp'])
    maxstatsvalue = np.array([df['maxstatsvalue']])

    sc = MinMaxScaler()
    data = sc.fit_transform(
        maxstatsvalue.transpose()
    )

    x, y = sliding_windows(data, seq_length)

    train_size = int(len(y) * 0.67)
    test_size = len(y) - train_size

    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))

    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

    return (train_size, test_size, dataX, dataY, trainX, trainY, testX, testY)


def train(
    trainX: torch.Tensor,
    trainY: torch.Tensor,
    num_epochs=40,
    lr=0.01,
    input_size=1,
    hidden_size=16,
    num_layers=1,
    num_classes=1,
):
    model = LSTM(num_classes, input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for _ in (prog := tqdm(range(num_epochs))):
        outputs: torch.Tensor = model(trainX)
        optimizer.zero_grad()

        loss: torch.Tensor = criterion(outputs.squeeze(), trainY)
        loss.backward()

        optimizer.step()
        prog.set_description(f'Loss: {loss.item()}')

    return model


def validate(valX: torch.Tensor, valY: torch.Tensor, model: LSTM):
    model.eval()
    predict = model(valX).data.numpy()
    real = valY.data.numpy()

    print(f'MAPE: {mean_absolute_percentage_error(real, predict)}')
    print(f'RMSE: {root_mean_squared_error(real, predict)}')


def predict(model: LSTM):
    model.eval()
