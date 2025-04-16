import torch
import torch.nn as nn
import math

from typing import Any, Literal, Optional, Union

### Dlinear Section
class moving_avg(torch.nn.Module):
    def __init__(self, ker_size, stride):
        super(moving_avg, self).__init__()
        self.ker_size = ker_size
        self.avg = nn.AvgPool1d(kernel_size=ker_size, stride=stride)
    
    def forward(self, x):
        # x shape: [B, S, C]
        front = x[:, 0:1, :].repeat(1, (self.ker_size-1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.ker_size-1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)

class series_decomp(torch.nn.Module):
    def __init__(self, ker_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(ker_size, stride=1)
    
    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return moving_mean, residual 

class multi_series_decomp(torch.nn.Module):
    def __init__(self, feature_ker_size):
        super(multi_series_decomp, self).__init__()
        self.moving_avgs = nn.ModuleList([moving_avg(ker_size, stride=1) \
                                          for ker_size in feature_ker_size])
    def forward(self, x):
        trends = []
        for i , pool in enumerate(self.moving_avgs):
            trend = pool(x[:, :, i:i+1])
            trends.append(trend)
        trends = torch.cat(trends, dim=2)
        residual = x - trends
        return trends, residual 
    
class DLinear_with_emb(nn.Module):
    def __init__(self, num_classes, input_size, 
                 seq_length, dropout_rate, ker_size=10, individual=True):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size 
        self.seq_length = seq_length
        self.dropout_rate = dropout_rate
        self.individual = individual 
        
        self.decomp = series_decomp(ker_size)
        self.Linear_Seaonal = nn.ModuleList()
        self.Linear_Trend = nn.ModuleList()
        if self.individual:
            for i in range(self.input_size):
                self.Linear_Trend.append(nn.Linear(self.seq_length, self.num_classes))
                self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_length)\
                                                           *torch.ones([self.num_classes, self.seq_length]))
                self.Linear_Seaonal.append(nn.Linear(self.seq_length, self.num_classes))
                self.Linear_Seaonal[i].weight = nn.Parameter((1/self.seq_length)\
                                                           *torch.ones([self.num_classes, self.seq_length]))
        else:
            self.Linear_Trend = nn.Linear(self.seq_length, self.num_classes)
            self.Linear_Trend.weight = nn.Parameter((1/self.seq_length)\
                                                           *torch.ones([self.num_classes, self.seq_length]))
            self.Linear_Seaonal = nn.Linear(self.seq_length, self.num_classes)
            self.Linear_Seaonal.weight = nn.Parameter((1/self.seq_length)\
                                                           *torch.ones([self.num_classes, self.seq_length]))
            
        self.fc1 =  nn.Linear(self.input_size, self.num_classes)
        self.fc2 =  nn.Linear(self.input_size, self.num_classes)
        
        self.d_emb = 12
        self.periodic = PeriodicEmbeddings(n_features = self.input_size, d_embedding=self.d_emb, 
                                           n_frequencies=self.d_emb*2 ,lite=True)
        self.emb_linear = nn.Linear(self.d_emb, 1)
        
    def forward(self, x, weight=None):
        # x: shape [B, S, C]
        # B is Batch size
        # S is history len
        # C is input feature size. e.g.,  EV, SMP, PC
        # weight is trend weight
        B, S, C = x.size() 
        x = x.reshape(-1, C)
        emb = self.periodic(x)
        # shape: [B*S, C, d_emb]
        x = self.emb_linear(emb)
        x = x.squeeze(-1)
        x = x.reshape(B, S, C)
        # shape: [B, S, C]
        
        trend_init, seasonal_init = self.decomp(x)
        # shape: [B, S, C]
       
        if weight is not None:
            plus_trend_signal = trend_init.mean(dim=1, keepdim=True).abs()
            weight = torch.tensor(weight, dtype=x.dtype).to(x.device)
            trend_init = trend_init + plus_trend_signal * weight[None, None, :]
            
        trend_init, seasonal_init = trend_init.permute(0, 2, 1), seasonal_init.permute(0, 2, 1)
        # shape: [B, C, S]
        trend_out = torch.zeros([trend_init.size(0), C, self.num_classes]
                                , dtype=trend_init.dtype).to(trend_init.device)
        seasonal_out = torch.zeros([seasonal_init.size(0), C, self.num_classes]
                                , dtype=seasonal_init.dtype).to(seasonal_init.device)
        
        if self.individual:
            for i in range(self.input_size):
                trend_out[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :]) 
                seasonal_out[:, i, :] = self.Linear_Seaonal[i](seasonal_init[:, i, :])
        else:
            trend_out = self.Linear_Trend(trend_init)
            seasonal_out = self.Linear_Seaonal(seasonal_init)
            
        trend_out = trend_out.permute(0, 2, 1)
        #shape: [B, S, C]
        seasonal_out = seasonal_out.permute(0, 2, 1)
        trend_flatten = trend_out.view(-1, self.input_size)
        seasonal_flatten = seasonal_out.view(-1, self.input_size)
        # out = nn.ReLU()(out)
        y_tout = self.fc1(trend_flatten)
        y_sout = self.fc2(seasonal_flatten)
        return y_tout + y_sout

### Embedding Section
def _check_input_shape(x: torch.Tensor, expected_n_features: int) -> None:
    if x.ndim < 1:
        raise ValeError("error x dim")
    if x.shape[-1] != expected_n_features:
        raise ValeError("error x dim")
    
class _Periodic(nn.Module):
    def __init__(self, n_features:int, k: int, sigma: float) -> None:
        super().__init__()
        self._sigma = sigma
        self.wegiht = nn.parameter.Parameter(torch.empty(n_features, k))
        self.reset_parameters()
    def reset_parameters(self):
        bound = self._sigma * 3
        nn.init.trunc_normal_(self.wegiht, 0.0, self._sigma, a=-bound, b=bound)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_input_shape(x, self.wegiht.shape[0])
        x = 2 * math.pi * self.wegiht * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x

class _NLinear(nn.Module):
    def __init__(self, n: int, in_features: int, out_features: int, bias: bool=True) -> None:
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.empty(n, in_features, out_features))
        self.bias = nn.parameter.Parameter(torch.empty(n, out_features)) if bias else None                                          
        self.reset_parameters()                             
    def reset_parameters(self):
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("error")
        assert x.shape[-(self.weight.ndim-1) :] == self.weight.shape[:-1]
        x = x.transpose(0, 1)
        x = x @ self.weight
        if self.bias is not None:
            x = x + self.bias
        return x

class PeriodicEmbeddings(nn.Module):
    def __init__(self, n_features: int, d_embedding: int=24, *, 
                n_frequencies: int=48,
                frequency_init_scale: float=0.01,
                activation: bool=True,
                lite: bool) -> None:
        super().__init__()
        self.periodic = _Periodic(n_features, n_frequencies, frequency_init_scale)
        self.linear: Union[nn.Linear, _NLinear]
        if lite:
            if not activation:
                raise ValueError("not defined actvaion func")
            self.linear = nn.Linear(2*n_frequencies, d_embedding)
        else:
            self.linear = _NLinear(n_features, 2*n_frequencies, d_embedding)
        self.activation = nn.ReLU() if activation else None
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.periodic(x)
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
