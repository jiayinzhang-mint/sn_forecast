import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from tsai.all import (TCN, Learner, TSDataLoaders, TSRegression, TSStandardize,
                      get_splits, get_ts_dls, mae, mape, rmse, ts_learner)

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


def sliding_windows_train(data: np.ndarray, window_size: int = 7*24*12):
    '''
    Sliding windows for training
    '''

    x = []
    y = []

    for i in range(len(data)-window_size-1-8*24*12):
        _x = data[i:(i+window_size)]
        x.append(_x)

        # y is the max of the respective next 7th day
        # e.g. when x is the data from [7/5~7/12), y is the max of [7/19,7/20)
        _y = np.max(data[i+window_size+7*24*12: i +
                    window_size+7*24*12+24*12])
        y.append(_y)

    return np.array(x), np.array(y)


def sliding_windows_predict(data: np.ndarray, window_size: int = 7*24*12):
    '''
    Sliding windows for prediction
    '''

    x = []

    for i in range(len(data)-window_size-1):
        _x = data[i:(i+window_size)]
        x.append(_x)

    return np.array(x)


def load_data(base_path: Path, ip: str, window_size: int = 7*24*12, device: torch.device = torch.device('cpu')):
    df = pd.read_csv(base_path.joinpath(
        f"{ip}.csv"), parse_dates=['timestamp'])
    maxstatsvalue = np.array([df['maxstatsvalue']])

    sc = MinMaxScaler()
    data = sc.fit_transform(
        maxstatsvalue.transpose()
    )

    x, y = sliding_windows_train(data, window_size)

    dataX = Variable(torch.Tensor(np.array(x)))
    dataX = dataX.reshape(dataX.shape[0], 1, dataX.shape[1])

    dataY = Variable(torch.Tensor(np.array(y)))

    splits = get_splits(dataY, valid_size=.3, stratify=True,
                        random_state=23, shuffle=True)
    tfms = [None, [TSRegression()]]
    batch_tfms = TSStandardize(by_sample=True, by_var=True)
    dls = get_ts_dls(dataX, dataY, splits=splits, tfms=tfms,
                     batch_tfms=batch_tfms, bs=128, device=device, seed=seed)

    return (sc, data, dataX, dataY, dls)


def train(
    dls: TSDataLoaders,
    epochs=50
):
    learner = ts_learner(dls, TCN, metrics=[
        mae, rmse, mape], seed=seed)
    lr = learner.lr_find()
    learner.fit_one_cycle(epochs, lr)

    return learner


def validate(valX: torch.Tensor, valY: torch.Tensor, learner: Learner):
    '''
    Validate the model with the validation set

    Args:
    - valX: np.ndarray
    - valY: np.ndarray
    - learner: Learner

    Returns:
    - float: MAE
    - float: RMSE
    - float: MAPE
    '''

    _, _, test_preds = learner.get_X_preds(
        valX, with_decoded=True)

    mae_score = mae(torch.tensor(test_preds).to('cpu'), valY.to('cpu'))
    rmse_score = rmse(torch.tensor(test_preds).to('cpu'), valY.to('cpu'))
    mape_score = mape(torch.tensor(test_preds).to('cpu'), valY.to('cpu'))

    return mae_score, rmse_score, mape_score


def predict(dataX: torch.Tensor, model: Learner, sc: MinMaxScaler):
    '''
    Predict the max usage of the next 7 days

    Args:
    - dataX: np.ndarray
    - model: Learner
    - sc: MinMaxScaler

    Returns:
    - np.ndarray: the max usage of the next 7 days (7,)
    '''

    # pick data with 24*12 as interval, 7 datapoints in total
    predictX = []
    for i in range(7):
        idx = -(6-i)*24*12-1
        data = dataX[idx]
        predictX.append(data)

    predictX = torch.from_numpy(np.array(predictX))
    _, _, test_preds = model.get_X_preds(
        predictX, with_decoded=True)

    return sc.inverse_transform(test_preds).squeeze()


def train_ip(ip: str, base_path: Path, epochs=50, device: torch.device = torch.device('cpu')):
    sc, data, _, _, dls = load_data(base_path, ip, device=device)
    learner = train(dls, epochs=epochs)
    valX, valY = dls.valid.one_batch()
    mae_score, rmse_score, mape_score = validate(
        valX, valY, learner=learner)

    dataX_predict = torch.from_numpy(sliding_windows_predict(data))
    dataX_predict = dataX_predict.reshape(
        dataX_predict.shape[0], 1, dataX_predict.shape[1])

    predict_res = predict(dataX_predict, learner, sc)

    return learner, mae_score, rmse_score, mape_score, predict_res


def train_ips(ips: list[str], base_path: Path, epochs=50, device: torch.device = torch.device('cpu')):
    res = pd.DataFrame(columns=['ip', 'mae_score',
                                'rmse_score', 'mape_score', '0', '1', '2', '3', '4', '5', '6'])
    for ip in ips:
        _, mae_score, rmse_score, mape_score, predict_res = train_ip(
            ip, base_path, epochs=epochs, device=device)
        res = pd.concat([res, pd.DataFrame(
            [[ip, mae_score, rmse_score, mape_score, *predict_res]],
            columns=res.columns)],
            ignore_index=True
        )
    return res
