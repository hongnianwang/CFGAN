import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Dropout, LayerNorm, Linear


def signed_matrix(matrix, dim=-1, eps=1e-8):                       
    """
    支持多批次的符号保留Softmax（Signed Softmax）
    输入: 
        matrix: 形状为 (B, S, S) 的张量
        dim: Softmax计算的维度（默认是最后一维）
        eps: 数值稳定性常数
    输出:
        符号保留的Softmax结果，形状与输入相同
    """
    # 屏蔽自环：对角线不参与 Softmax 归一化
    # （论文公式分母为 sum_{k != i} exp(|Corr_ik|)，自环既不在分子也不在分母）
    mask = torch.eye(matrix.size(-1), device=matrix.device, dtype=torch.bool)

    # 记录符号（对原始相关矩阵取符号，对角线随后被置零）
    sign_matrix = torch.sign(matrix)

    # 对绝对值进行Softmax（对角线置 -inf，exp 后为 0）
    abs_matrix = torch.abs(matrix).masked_fill(mask, float("-inf"))
    softmax_matrix = F.softmax(abs_matrix, dim=dim)

    # 恢复符号
    return softmax_matrix * sign_matrix

def compute_correlations(data):
    """
    data: [S, T, N]  S=batch, T=seq_len, N=nodes
    mode: "time" | "freq_mag" | "freq_real_imag"
    返回: [N, S, S] 或 ([N, S, S], [N, S, S])
    """
    S, T, N = data.shape
    X = data.permute(2, 0, 1)  # [N, S, T]

    X_centered = X - X.mean(dim=2, keepdim=True)
    cov = torch.matmul(X_centered, X_centered.transpose(1, 2)) / (T - 1)  # [N, S, S]
    std = X.std(dim=2, unbiased=True)
    std_matrix = std.unsqueeze(2) * std.unsqueeze(1) + 1e-8
    corr = cov / std_matrix
    return corr  # [N, S, S]



import torch
import torch.nn as nn
import torch.nn.functional as F
class NodeNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: [N, S, K]
        mean = x.mean(dim=1, keepdim=True)  # [N, 1, K]
        std = x.std(dim=1, keepdim=True) + self.eps  # [N, 1, K]
        x_norm = (x - mean) / std
        return x_norm, mean, std

    def denorm(self, x_norm, mean, std):
        return x_norm * std + mean


class CSFEM(nn.Module):
    def __init__(self):
        super().__init__()
        self.nodenorm_real = NodeNorm()
        self.nodenorm_imag = NodeNorm()

    def forward(self, x, high_matrix_time=None, high_matrix_freq=None,
                high_matrix_real=None, high_matrix_imag=None):
        S, T, N = x.shape
        x = x.permute(0,2,1).reshape(-1, T)
        x_complex = torch.fft.rfft(x, dim=-1, norm='ortho')
        x_complex = x_complex.reshape(S, N, -1).permute(1,0,2)
        x_real = x_complex.real
        x_imag = x_complex.imag

        x_real_normed, mean_real, std_real = self.nodenorm_real(x_real)
        x_imag_normed, mean_imag, std_imag = self.nodenorm_imag(x_imag)

        h_real = torch.matmul(high_matrix_time, x_real_normed)
        h_imag = torch.matmul(high_matrix_time, x_imag_normed)

        h_real = self.nodenorm_real.denorm(h_real, mean_real, std_real)
        h_imag = self.nodenorm_imag.denorm(h_imag, mean_imag, std_imag)

        complex_out = torch.complex(h_real, h_imag)
        x_complex = complex_out.permute(1,0,2).reshape(S*N, -1)
        out = torch.fft.irfft(x_complex, n=T, dim=-1, norm='ortho')
        out = out.reshape(S, N, T).permute(0,2,1)
        return out
#  -----------------------------
# Feature Extractor
# -----------------------------
class TDFFM(nn.Module):
    def __init__(self, d_fead, embed_dim=32, dropout=0.1, num_heads=4):
        super().__init__()
        self.complex_gnn = CSFEM()
        self.fc_1 = nn.Sequential(nn.Linear(d_fead, embed_dim), nn.GELU())
        self.fc_2 = nn.Sequential(nn.Linear(d_fead, embed_dim), nn.GELU())
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        origin_x = x
        corr_time = compute_correlations(x)
        high_matrix_time = signed_matrix(corr_time)
        out = self.complex_gnn(x, high_matrix_time=high_matrix_time)
        out = self.fc_1(out)
        x_proj = self.fc_2(origin_x)
        attn_output, _ = self.attn(query=out, key=x_proj, value=x_proj)
        fused = self.norm(x_proj + self.dropout(attn_output))
        return fused


class CFGAN(nn.Module):
    def __init__(self, d_feat, seq_len=60, hidden_size=128, num_layers=2, num_heads=4, embed_dim=32, dropout=0.1):
        super().__init__()
        self.d_feat = d_feat
        self.feature_extractor = TDFFM(
            d_fead=d_feat,
            embed_dim=embed_dim,
            dropout=dropout,
            num_heads=num_heads
        )
        self.alstm = ALSTMModel(
            d_feat=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type="LSTM",
        )
     
    def forward(self, x):
        x =  x.reshape(len( x), self.d_feat, -1)
        x=  x.permute(0, 2, 1)
        fused = self.feature_extractor(x)
        return self.alstm(fused)


# ALSTMModel is adapted from Microsoft Qlib (https://github.com/microsoft/qlib),
# Copyright (c) Microsoft Corporation, licensed under the MIT License.
class ALSTMModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU"):
        super().__init__()
        self.hid_size = hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self._build_model()
    

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module("act", nn.Tanh())
        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=1)
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size, out_features=int(self.hid_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size / 2), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, inputs):
        # inputs: [batch_size, input_size*input_day]
        rnn_out, _ = self.rnn(self.net(inputs))  # [batch, seq_len, num_directions * hidden_size]
        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1]
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        
        last_out =rnn_out[:, -1, :]
        out = torch.cat((last_out, out_att), dim=1)
        out = self.fc_out(out)  # [batch, seq_len, num_directions * hidden_size] -> [batch, 1]
        # return out
        return out[..., 0]