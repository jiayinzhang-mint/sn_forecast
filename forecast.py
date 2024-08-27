import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from tsai.all import (TCN, Learner, SaveModel, TSDataLoaders, TSRegression,
                      TSStandardize, get_splits, get_ts_dls, mae, mape, rmse,
                      ts_learner)

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


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


def load_data(base_path: Path,
              ip: str,
              window_size: int = 7*24*12,
              rolling=1,
              device: torch.device = torch.device('cpu'),
              ):

    if ip.endswith('*'):
        ips = [f.stem for f in base_path.glob(f"{ip[:-1]}*.csv")]
        print('selected ips:', ips)

        # load all csv files with the prefix
        df = pd.concat([pd.read_csv(f, parse_dates=['timestamp'])
                        for f in base_path.glob(f"{ip[:-1]}*.csv")])
    else:
        df = pd.read_csv(base_path.joinpath(
            f"{ip}.csv"), parse_dates=['timestamp'])
        ips = [ip]

    # rolling
    df_roll = df['maxstatsvalue'].rolling(rolling).mean()
    df_roll = df_roll.dropna()
    df_roll = df_roll.reset_index(drop=True)

    maxstatsvalue = np.array([df_roll])

    sc = MinMaxScaler()
    data = sc.fit_transform(
        maxstatsvalue.transpose()
    )

    x, y = sliding_windows_train(data, window_size)

    dataX = Variable(torch.Tensor(np.array(x)))
    dataX = dataX.reshape(dataX.shape[0], 1, dataX.shape[1])

    dataY = Variable(torch.Tensor(np.array(y)))

    splits = get_splits(dataY, valid_size=.3, stratify=True,
                        random_state=23, shuffle=True, show_plot=False)
    tfms = [None, [TSRegression()]]
    batch_tfms = TSStandardize(by_sample=True, by_var=True)
    dls = get_ts_dls(dataX, dataY, splits=splits, tfms=tfms,
                     batch_tfms=batch_tfms, bs=128, device=device, seed=seed)

    return (sc, data, dataX, dataY, dls, ips)


def train(
    dls: TSDataLoaders,
    epochs=50,
    show_plot=False,
    save_model_path: Path | None = None,
):
    cbs = []
    if save_model_path:
        cbs.append(SaveModel(fname=str(save_model_path)))

    learner = ts_learner(dls, TCN, metrics=[
        mae, rmse, mape], seed=seed)
    lr = learner.lr_find(show_plot=show_plot)
    learner.fit_one_cycle(
        epochs, lr, cbs=cbs)

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

    mae_score: torch.Tensor = mae(torch.tensor(
        test_preds).to('cpu'), valY.to('cpu'))
    rmse_score: torch.Tensor = rmse(
        torch.tensor(test_preds).to('cpu'), valY.to('cpu'))  # type: ignore
    mape_score: torch.Tensor = mape(
        torch.tensor(test_preds).to('cpu'), valY.to('cpu'))

    return mae_score.item(), rmse_score.item(), mape_score.item()


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
        predictX.append(data.numpy().tolist())

    predictX = torch.from_numpy(np.array(predictX))
    _, _, test_preds = model.get_X_preds(
        predictX, with_decoded=True)

    return sc.inverse_transform(test_preds).squeeze()


def train_ip(ip: str, base_path: Path, epochs=50, rolling=1,
             device: torch.device = torch.device('cpu'),
             save_model_dir: Path | None = None,
             res: pd.DataFrame | None = None,):
    '''
    Train the model for a specific ip or a set of ips with the same prefix

    Args:
    - ip: a specific ip or a prefix with * (e.g. 1.2.3.4 or 10.0.0.*)
    - base_path: dataset path with csv files for each ip
    - epochs: number of epochs for training
    - rolling: rolling window size
    - device: torch.device
    - save_model_dir: save the model to the specified directory
    '''

    _save_model_dir = None
    if save_model_dir:
        _save_model_dir = save_model_dir.joinpath(ip)

    sc, data, _, _, dls, ips = load_data(
        base_path=base_path, ip=ip, rolling=rolling, device=device)
    learner = train(dls, epochs=epochs,
                    save_model_path=_save_model_dir)
    valX, valY = dls.valid.one_batch()
    mae_score, rmse_score, mape_score = validate(
        valX, valY, learner=learner)

    failed_ips = []

    if res is None:
        res = pd.DataFrame(columns=['ip', 'mae_score',
                                    'rmse_score', 'mape_score', '0', '1', '2', '3', '4', '5', '6'])

    # load data for prediction
    for ip_item in ips:
        try:
            _, data, _, _, _, _ = load_data(
                base_path=base_path, ip=ip_item, rolling=rolling, device=device)

            dataX_predict = torch.from_numpy(sliding_windows_predict(data))
            dataX_predict = dataX_predict.reshape(
                dataX_predict.shape[0], 1, dataX_predict.shape[1])

            predict_res = predict(dataX_predict, learner, sc)

            res = pd.concat([res, pd.DataFrame([[ip, mae_score, rmse_score,
                            mape_score, *predict_res]], columns=res.columns)], ignore_index=True)
            res.to_csv(output_path, index=False)
        except Exception as e:
            print(f"Failed to predict ip {ip}, error: {e}")
            failed_ips.append(ip_item)

    if len(failed_ips) > 0:
        print('failed ips: ', failed_ips)

    return learner, mae_score, rmse_score, mape_score, res


def find_ip_prefixes(base_path: Path, segments: int = 2):
    '''
    Find all ip prefixes with the same prefix

    Args:
    - base_path: dataset path with csv files for each ip
    - segments: number of segments to consider (e.g. 3 for 10.0.0.*)
    '''

    ips = [f.stem for f in base_path.glob('*.csv')]
    prefixes = [('.'.join(ip.split('.')[:segments]) + '*') for ip in ips]

    return list(set(prefixes))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--base_path', type=str,
                           default='./data/dfyj/key1_20240618_20240718', required=True)
    argparser.add_argument('--ip', type=str, default='',
                           help='ip or ip prefix with *. if not provided, predict all ips')
    argparser.add_argument('--segments', type=int, default=2,
                           help='number of segments to consider (e.g. 3 for 10.0.0.*)')
    argparser.add_argument('--epochs', type=int, default=50)
    argparser.add_argument('--rolling', type=int, default=1)
    argparser.add_argument('--output_path', type=str, required=True)
    argparser.add_argument('--save_model_dir', default='', type=str)

    args = argparser.parse_args()
    base_path = Path(args.base_path)
    ip = str(args.ip)
    segments = int(args.segments)
    epochs = int(args.epochs)
    rolling = int(args.rolling)
    output_path = Path(args.output_path)
    save_model_dir = Path(args.save_model_dir)

    if save_model_dir:
        Path('./models').joinpath(save_model_dir).mkdir(exist_ok=True)

    if ip != '':
        _, _, _, _, res = train_ip(ip, base_path, epochs=epochs, rolling=rolling, device=device,
                                   save_model_dir=save_model_dir)
    elif segments:
        prefixes = find_ip_prefixes(base_path, segments)
        print('prefixes: ', prefixes)
        res = pd.DataFrame(columns=['ip', 'mae_score',
                                    'rmse_score', 'mape_score', '0', '1', '2', '3', '4', '5', '6'])
        for prefix in prefixes:
            _, _, _, _, _res = train_ip(prefix, base_path, epochs=epochs, rolling=rolling, device=device,
                                        save_model_dir=save_model_dir, res=res)
            res = pd.concat([res, _res], ignore_index=True)
            print('-----------------')

    # remove score columns and add a suffix to the original filename
    res.drop(columns=['mae_score', 'rmse_score', 'mape_score']).to_csv(
        output_path.with_name(output_path.stem + '_predict.csv'), index=False)
