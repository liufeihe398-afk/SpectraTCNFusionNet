
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from scipy.stats import zscore
import os
import torch.nn.functional as F
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
def Compute_error(actuals, predictions, history=None):
    actuals = actuals.ravel()  
    predictions = predictions.ravel()  
    error = {}
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    error['RMSE'] = rmse

    mae = np.mean(np.abs(actuals - predictions))
    error['MAE'] = mae
    epsilon = 1e-8  
    mape = np.mean(np.abs((actuals - predictions) / (actuals + epsilon))) * 100
    error['MAPE'] = mape


    # smape = np.mean(2 * np.abs(actuals - predictions) / (np.abs(actuals) + np.abs(predictions))) * 100
    # error['SMAPE'] = smape

    # rmsle = np.sqrt(np.mean((np.log1p(actuals) - np.log1p(predictions)) ** 2))
    # error['RMSLE'] = rmsle


    mean_actuals = np.mean(actuals)
    ss_total = np.sum((actuals - mean_actuals) ** 2) 
    ss_residual = np.sum((actuals - predictions) ** 2) 
    r2 = 1 - (ss_residual / ss_total)
    error['R²'] = r2
    if history is not None:
        history = history.ravel() 
        
        baseline_mae = np.mean(np.abs(history[1:] - history[:-1]))  
        mase = mae / baseline_mae
        error['MASE'] = mase
    return error




def save_error_to_file(error, file_name='evaluation_results.txt', line_name="line_name"):
    with open(file_name, 'a+',encoding='utf-8') as f:

        f.write(f"{line_name}")
        for metric, value in error.items():
            f.write(f" {metric}:  {value:.6f}")
        f.write("\n")


class ComplexReLU(nn.Module):
    def forward(self, x):
        real = torch.relu(x.real)
        imag = torch.relu(x.imag)
        return torch.complex(real, imag)

class ComplexDropout(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ComplexDropout, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        real = self.dropout(x.real)
        imag = self.dropout(x.imag)
        return torch.complex(real, imag)

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class FrequencySamplingRefinementModule(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, dropout):
        super(FrequencySamplingRefinementModule, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len
        self.dominance_freq = int(self.seq_len / 2 + 1)
        self.pre_freq = int((self.seq_len + self.pred_len) / 2 + 1)
        self.freq_upsampler = nn.Linear(self.dominance_freq, self.pre_freq).to(torch.cfloat)
        self.freq_upsampler1 = nn.Linear(self.pre_freq, self.pre_freq).to(torch.cfloat)
        self.dropout = dropout
        self.complex_relu = ComplexReLU()
        self.complex_dropout = ComplexDropout(dropout_rate=self.dropout)
    def forward(self, x):
        spec_x = torch.fft.rfft(x, dim=1)                       # [B, F, C] complex
        spec_x_up = self.freq_upsampler(spec_x.permute(0, 2, 1)).permute(0, 2, 1)
        spec_x_up = self.complex_relu(spec_x_up)
        spec_x_up = self.complex_dropout(spec_x_up)
        spec_x_up = self.freq_upsampler1(spec_x_up.permute(0,2,1)).permute(0, 2, 1)
        x_up = torch.fft.irfft(spec_x_up, dim=1)
        x_up = x_up * self.length_ratio
        # x_up = (x_up) * torch.sqrt(x_var) + x_mean
        return x_up


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, T, C] -> conv expects [B, C, T]
        out = self.network(x.transpose(1,2))   # [B, C_out, T]
        return out.transpose(1,2)              # [B, T, C_out]

class StarAttention(nn.Module):
    def __init__(self, feature, d_core):
        super(StarAttention, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """
        # self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(feature, d_core)
        self.gen3 = nn.Linear(feature + d_core, feature)
        self.gen4 = nn.Linear(feature, feature)

    def forward(self, input, *args, **kwargs):
        batch_size, series, feature = input.shape
        combined_mean = self.gen2(input)
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1) 
            ratio = ratio.reshape(-1, series)
            indices = torch.multinomial(ratio, 1) 
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, series, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, series, 1)
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        return combined_mean_cat, None



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = 0.02
        # RevIN
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=True)
        self.hidden_size = configs.hidden_size

        self.w = nn.Parameter(self.scale * torch.randn(1, self.seq_len))

        self.FrequencyEnhancementBlock = FrequencySamplingRefinementModule(self.seq_len, self.pred_len, configs.enc_in, 0.1)
        tcn_out_ch = configs.enc_in

        self.attention = StarAttention(1,8)
        self.alpha = nn.Parameter(torch.full((1, self.seq_len, 1), 1), requires_grad=False)
        self.tcn = TemporalConvNet(num_inputs=configs.enc_in, num_channels=[16, tcn_out_ch], kernel_size=3, dropout=0.1)

        self.fc = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),


            nn.Linear(self.hidden_size, self.pred_len)
        )

    def circular_convolution(self, x, w):
        # x: [B, C, T] in your call it was [B, T, C] but we used earlier [B, feature, time] convention
        # Here we stick with your original implementation in forward: expects x [B, feature, T] and w [1, seq_len]
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(w, dim=1, norm='ortho')
        y = x * w
        out = torch.fft.irfft(y, n=self.seq_len, dim=2, norm="ortho")
        return out

    def forward(self, x, x_mark_batch=None, dec_inp=None, y_mark_batch=None):
        # x: [B, T, C]

        Frequency = self.revin_layer(x, 'norm')
        x_norm = Frequency
        Frequency = Frequency.permute(0, 2, 1)
        Frequency = self.circular_convolution(Frequency, self.w.to(Frequency.device))  # [B, C, T]
        Frequency = Frequency.permute(0, 2, 1)

        Frequency = Frequency+x_norm
        Frequency = self.FrequencyEnhancementBlock(Frequency)

        time, attn = self.attention(x)
        time = self.tcn(time)

        fused = self.alpha * Frequency + (1 - self.alpha) * time

        x_fc = fused.permute(0, 2, 1)   # [B, C, T]
        x_fc = self.fc(x_fc)                 # [B, hidden_size, pred_len]
        x_fc = x_fc.permute(0, 2, 1)         # [B, pred_len, hidden_size]  

        out = x_fc

        if out.shape[-1] != 1:
            proj = nn.Linear(out.shape[-1], 1).to(out.device)
            out = proj(out)   # [B, pred_len, 1]

        out = out  # already [B, pred_len, 1]

        # denorm
        out = out.permute(0, 2, 1)  # [B, 1, pred_len]  
        out = self.revin_layer(out, 'denorm')
        out = out.permute(0, 2, 1)  # [B, pred_len, 1]
        return out




def load_and_normalize_data(path, input_features, output_features):
    data = pd.read_csv(path)
    data1 = data
    data['DateUTC'] = pd.to_datetime(data['DateUTC'], format='%d-%m-%Y %H:%M')
    data['month'] = data['DateUTC'].dt.month / 12.0
    data['day'] = data['DateUTC'].dt.day / 31.0
    data['weekday'] = data['DateUTC'].dt.weekday / 6.0
    data['hour'] = data['DateUTC'].dt.hour / 23.0
    time_features = data[['month', 'day', 'weekday', 'hour']].values.astype(np.float32)
    data = data.dropna(subset=input_features)
    for col in input_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data[data[input_features] < 1e7]
    data[input_features] = data[input_features].astype(float)
    data.dropna(subset=input_features, inplace=True)

    z_thresh = 3.0
    z_scores = np.abs(zscore(data[input_features]))
    mask = (z_scores < z_thresh).all(axis=1)
    data = data[mask]
    # plt.plot(data[input_features].astype(np.float32))
    # plt.show()
    feature_scaler = MinMaxScaler(feature_range=(-1, 1))
    features = feature_scaler.fit_transform(data[input_features]).astype(np.float32)
    target_scaler = {}
    target_scaled = []

    for col in output_features:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = scaler.fit_transform(data[[col]]).astype(np.float32)
        target_scaled.append(scaled)
        target_scaler[col] = scaler
    target = np.concatenate(target_scaled, axis=1)




    return features, time_features,target, feature_scaler, target_scaler,data1




def split_data(dataset, time_features, lookback=24, label_len=24, predict_step=1, step=1, train_ratio=0.8):
    total_len = label_len + predict_step
    num_samples = (len(dataset) - lookback - total_len) // step + 1

    x_arr = np.zeros((num_samples, lookback, dataset.shape[1]), dtype=np.float32)
    target_arr = np.zeros((num_samples, total_len, dataset.shape[1]), dtype=np.float32)
    time_mark = np.zeros((num_samples, lookback, time_features.shape[1]), dtype=np.float32)
    y_mark = np.zeros((num_samples, total_len, time_features.shape[1]), dtype=np.float32)

    for i in range(0, num_samples):
        start_idx = i * step
        x_arr[i] = dataset[start_idx : start_idx + lookback]
        target_arr[i] = dataset[start_idx + lookback -label_len: start_idx + lookback + predict_step]

        time_mark[i] = time_features[start_idx : start_idx + lookback]
        y_mark[i] = time_features[start_idx + lookback- label_len : start_idx + lookback + predict_step]

    train_size = int(num_samples * train_ratio)

    x_train = x_arr[:train_size]
    y_train = target_arr[:train_size]
    x_test = x_arr[train_size:]
    y_test = target_arr[train_size:]

    x_time_train = time_mark[:train_size]
    x_time_test = time_mark[train_size:]
    y_time_train = y_mark[:train_size]
    y_time_test = y_mark[train_size:]

    return x_train, y_train, x_test, y_test, x_time_train, x_time_test, y_time_train, y_time_test


from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, x_mark=None,y_mark=None):
        self.x = x
        self.y = y
        self.x_mark = x_mark if x_mark is not None else torch.zeros_like(x[:, :, :4])  # 占位
        self.y_mark = y_mark if y_mark is not None else torch.zeros((x.shape[0], y.shape[1], self.x_mark.shape[-1]))
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.x_mark[idx],self.y_mark[idx]

from torch.utils.data import DataLoader

def create_dataloaders(x_train, y_train, x_test, y_test, x_mark_train,x_mark_test, y_mark_train, y_mark_test, batch_size=64):
    train_dataset = TimeSeriesDataset(x_train, y_train,x_mark_train,y_mark_train)
    test_dataset = TimeSeriesDataset(x_test, y_test,x_mark_test,y_mark_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(dataset):

    # set_random_seed(42)
    dataset = dataset
    model_names =[
        'SpectraTCNFusionNet'
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=dataset)
    parser.add_argument('--vision', type=bool, default=True)
    parser.add_argument('--input_features', type=list, default=['Value'])
    parser.add_argument('--output_features', type=list, default=['Value'])
    # parser.add_argument('--input_features',type=list,default=['KW',])
    # parser.add_argument('--output_features',type=list,default=['KW'])
    # parser.add_argument('--window_size', type=int, default=96)
    # parser.add_argument('--label_len',type = int, default=48)
    # parser.add_argument('--predict_step', type=int, default=24)
    # parser.add_argument('--window_size', type=int, default=196)
    # parser.add_argument('--label_len',type = int, default=12)
    # parser.add_argument('--predict_step', type=int, default=1)
    parser.add_argument('--window_size', type=int, default=24)
    parser.add_argument('--label_len',type = int, default=12)
    parser.add_argument('--predict_step', type=int, default=1)

    parser.add_argument('--train_test_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--random_state', type=int, default=34)

    # parser.add_argument('--num_epochs', type=int, default=200)

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4, help='Adam learning rate')
    args = parser.parse_args(args=[])
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    features,time_features, target, feature_scaler, target_scaler,data = load_and_normalize_data(
        args.path, args.input_features, args.output_features)
    x_train, y_train, x_test, y_test,x_time_train,x_time_test,y_time_train,y_time_test = split_data(features,time_features, lookback=args.window_size, label_len=args.label_len,predict_step=args.predict_step)
    print("x_train",x_train.shape)
    print("y_train",y_train.shape)
    print("x_test",x_test.shape)


    train_loader, test_loader = create_dataloaders(
        x_train, y_train, x_test, y_test, x_time_train,x_time_test,y_time_train,y_time_test,args.batch_size)


    results ={"Actual load":None}
    for model_name in model_names:
        if model_name == 'TexFilter':
            pass

        elif model_name == 'SpectraTCNFusionNet':
            # from model1.PaiFilter_TCN6 import Model

            class Configs:
                def __init__(self):
                    self.seq_len = args.window_size
                    self.pred_len = args.predict_step
                    self.enc_in = len(args.input_features)
                    self.enc_out =len(args.output_features)
                    self.hidden_size = 256

            configs = Configs()
            model = Model(configs)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


        train_MSE_hist = []
        test_MSE_hist = []

        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0.0
            task_train_losses ={name:0 for i,name in enumerate(args.output_features)}

            for x_batch, y_batch,x_mark_batch,y_mark_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)  # shape: [batch, 3]
                x_mark_batch = x_mark_batch.to(device)
                y_mark_batch =y_mark_batch.to(device)
                # print("x_batch",x_batch.shape)
                # print("x_mark_batch",x_mark_batch.shape)
                dec_inp = torch.zeros_like(y_batch[:, -args.predict_step:, :])
                dec_inp = torch.cat([y_batch[:, :args.label_len, :], dec_inp], dim=1).to(device)

                # print("x_batch",x_batch.shape)
                # print("x_mark_batch",x_mark_batch.shape)
                # print("y_batch", dec_inp.shape)
                # print("y_mark_batch",y_mark_batch.shape)
                # print("dec_inp",dec_inp.shape)
                # cycle_index = torch.randint(0, 24, (x_batch.shape[0],)) .to(device) # 每个样本的周期位置
                # x_mark_batch = cycle_index
                # print(model)
                preds =model(x_batch,x_mark_batch,dec_inp,y_mark_batch)
                # print(type(preds))

                # print("preds",preds.shape)
                # preds = model(x_batch)
                y_batch =y_batch[:, -args.predict_step:, :]

                y_true = {name: y_batch[:, :, i] for i, name in enumerate(args.output_features)}
                preds = {name: preds[:, :, i] for i, name in enumerate(args.output_features)}

                losses = {task: F.mse_loss(preds[task], y_true[task]) for task in preds}
                loss = sum(losses.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * x_batch.size(0)
                for task in task_train_losses:
                    task_train_losses[task] += losses[task].item() * x_batch.size(0)

            avg_loss = epoch_loss / len(train_loader.dataset)
            scheduler.step()
            train_MSE_hist.append(avg_loss)
            print(f"[Epoch {epoch + 1}] Train Loss: {avg_loss:.6f}")

            # ---------- Test ----------
            model.eval()
            test_loss = 0.0
            # task_test_losses = {'KW': 0, 'HTmmBTU': 0, 'CHWTON': 0}
            # all_preds = {'KW': [], 'HTmmBTU': [], 'CHWTON': []}
            # all_labels = {'KW': [], 'HTmmBTU': [], 'CHWTON': []}
            task_test_losses = {name: 0 for i, name in enumerate(args.output_features)}
            all_preds = {name:[]for i,name in enumerate(args.output_features)}
            all_labels = {name:[]for i,name in enumerate(args.output_features)}
            with torch.no_grad():
                for x_batch, y_batch ,x_mark_batch,y_mark_batch in test_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)  # shape: [batch, 3]
                    x_mark_batch = x_mark_batch.to(device)
                    y_mark_batch = y_mark_batch.to(device)

                    # print("x_batch",x_batch.shape)
                    # print("x_mark_batch",x_mark_batch.shape)
                    dec_inp = torch.zeros_like(y_batch[:, -args.predict_step:, :])
                    dec_inp = torch.cat([y_batch[:, :args.label_len, :], dec_inp], dim=1).to(device)
                    # cycle_index = torch.randint(0, 24, (x_batch.shape[0],)).to(device)  # 每个样本的周期位置
                    # x_mark_batch = cycle_index

                    preds =model(x_batch,x_mark_batch,dec_inp,y_mark_batch)

                    y_batch = y_batch[:, -args.predict_step:, :]

                    y_true = {name: y_batch[:, :, i] for i, name in enumerate(args.output_features)}
                    preds = {name: preds[:, :, i] for i, name in enumerate(args.output_features)}

                    losses = {task: F.mse_loss(preds[task], y_true[task]) for task in preds}

                    loss = sum(losses.values())
                    test_loss += loss.item() * x_batch.size(0)

                    for task in all_preds:
                        all_preds[task].append(preds[task].cpu())
                        all_labels[task].append(y_true[task].cpu())
                        task_test_losses[task] += losses[task].item() * x_batch.size(0)

            test_avg_loss = test_loss / len(test_loader.dataset)
            test_MSE_hist.append(test_avg_loss)

            print(f"[Epoch {epoch + 1}] Test Loss: {test_avg_loss:.6f}")

            # from compute_error import compute_error
            for task in args.output_features:
                y_pred = torch.cat(all_preds[task], dim=0).numpy().reshape(-1, 1)
                y_true = torch.cat(all_labels[task], dim=0).numpy().reshape(-1, 1)

                y_pred = target_scaler[task].inverse_transform(y_pred).flatten()
                y_true = target_scaler[task].inverse_transform(y_true).flatten()

                error = Compute_error(y_pred, y_true)
                print(f"{task} : {error}")

        plt.plot(train_MSE_hist, label="Train MSE")
        plt.plot(test_MSE_hist, '--', label="Test MSE")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid(True)
        plt.show()

        model.eval()
        test_loss = 0.0
        all_preds = {name: [] for i, name in enumerate(args.output_features)}
        all_labels = {name: [] for i, name in enumerate(args.output_features)}

        with torch.no_grad():
            for x_batch, y_batch, x_mark_batch, y_mark_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)  # shape: [batch, 3]
                x_mark_batch = x_mark_batch.to(device)
                y_mark_batch = y_mark_batch.to(device)

                # print("x_batch",x_batch.shape)
                # print("x_mark_batch",x_mark_batch.shape)
                dec_inp = torch.zeros_like(y_batch[:, -args.predict_step:, :])
                dec_inp = torch.cat([y_batch[:, :args.label_len, :], dec_inp], dim=1).to(device)
                # cycle_index = torch.randint(0, 24, (x_batch.shape[0],)).to(device)
                # x_mark_batch = cycle_index

                preds = model(x_batch, x_mark_batch, dec_inp, y_mark_batch)

                y_batch = y_batch[:, -args.predict_step:, :]



                # y_batch = y_batch.squeeze(1)  # shape: [batch, 3]
                # y_pred = y_pred.squeeze(1)  # shape: [batch, 3]
                y_true = {name: y_batch[:, :, i] for i, name in enumerate(args.output_features)}
                preds = {name: preds[:, :, i] for i, name in enumerate(args.output_features)}

                loss = sum(F.mse_loss(preds[k], y_true[k]) for k in preds) / len(preds)
                test_loss += loss.item() * x_batch.size(0)


                for k in preds:
                    all_preds[k].append(preds[k].cpu())
                    all_labels[k].append(y_true[k].cpu())


        for k in all_preds:
            all_preds[k] = torch.cat(all_preds[k], dim=0).numpy()
            all_labels[k] = torch.cat(all_labels[k], dim=0).numpy()

        for k in all_preds:
            pred = target_scaler[k].inverse_transform(all_preds[k].reshape(-1, 1)).flatten()
            true = target_scaler[k].inverse_transform(all_labels[k].reshape(-1, 1)).flatten()
            plt.figure(figsize=(10, 4))
            plt.plot(pred[-100:], label='Predicted')
            plt.plot(true[-100:], label='True')
            plt.title(f'{k} - Predicted vs True')
            plt.legend()
            plt.tight_layout()
            plt.show()

            if results["Actual load"] is None:
                results["Actual load"] =true
            results[f"{model_name}"] = pred
            from compute_error import compute_error, save_error_to_file
            error = compute_error(pred, true)

            from pathlib import Path
            filename = (
                    Path(dataset).stem
                    + " " + str(k)
                    + " " + str(args.window_size)
                    + " " + str(args.predict_step)
                    + " " + model_name
                    + " " + str(args.num_epochs)
                    + " : "
            )
            print(model_name)

            save_error_to_file(error, file_name=r"E:\BaiduNetdiskDownload\B913-KAN+Transformer时间序列预测完整\time_series\timeseries\examples\model1\ablation1\ablation.txt", line_name=filename)



    # df = pd.DataFrame(results)
    # df.insert(0, "DateUTC", data['DateUTC'].iloc[-len(df):].values)
    # csv_path = (
    #         r"E:\BaiduNetdiskDownload\B913-KAN+Transformer时间序列预测完整\time_series\timeseries\examples\对比实验\国家实验\数据集\真实数据集"
    #         + "\\" + Path(dataset).stem + ".csv"
    # )
    # df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    # print(f"已保存所有模型结果到 {csv_path}")

if __name__ == '__main__':
    dataset_paths = [
        r"E:\BaiduNetdiskDownload\B913-KAN+Transformer时间序列预测完整\time_series\timeseries\examples\dataset\countrydata\DE.csv",
        r"E:\BaiduNetdiskDownload\B913-KAN+Transformer时间序列预测完整\time_series\timeseries\examples\dataset\countrydata\FR.csv",
        r"E:\BaiduNetdiskDownload\B913-KAN+Transformer时间序列预测完整\time_series\timeseries\examples\dataset\countrydata\GB.csv",
        r"E:\BaiduNetdiskDownload\B913-KAN+Transformer时间序列预测完整\time_series\timeseries\examples\dataset\countrydata\PL.csv"
    ]
    for dataset_path in dataset_paths:
        main(dataset_path)

