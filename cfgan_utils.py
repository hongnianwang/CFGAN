import torch
import pandas as pd
import numpy as np
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def mse(pred, label):
    """
    计算均方误差（Mean Squared Error）。

    参数：
    - pred：预测值
    - label：真实值

    返回：
    - 均方误差
    """
    loss = (pred - label) ** 2
    return torch.mean(loss)


def mae(pred, label):
    """
    计算平均绝对误差（Mean Absolute Error）。

    参数：
    - pred：预测值
    - label：真实值

    返回：
    - 平均绝对误差
    """
    loss = (pred - label).abs()
    return torch.mean(loss)


def cal_cos_similarity(x, y):
    """
    计算余弦相似度。

    参数：
    - x：输入向量
    - y：输入向量

    返回：
    - 余弦相似度
    """
    xy = x.mm(torch.t(y))
    x_norm = torch.sqrt(torch.sum(x * x, dim=1)).reshape(-1, 1)
    y_norm = torch.sqrt(torch.sum(y * y, dim=1)).reshape(-1, 1)
    cos_similarity = xy / x_norm.mm(torch.t(y_norm))
    cos_similarity[cos_similarity != cos_similarity] = 0
    return cos_similarity


def cal_convariance(x, y):
    """
    计算协方差。

    参数：
    - x：输入向量
    - y：输入向量

    返回：
    - 协方差
    """
    e_x = torch.mean(x, dim=1).reshape(-1, 1)
    e_y = torch.mean(y, dim=1).reshape(-1, 1)
    e_x_e_y = e_x.mm(torch.t(e_y))
    x_extend = x.reshape(x.shape[0], 1, x.shape[1]).repeat(1, y.shape[0], 1)
    y_extend = y.reshape(1, y.shape[0], y.shape[1]).repeat(x.shape[0], 1, 1)
    e_xy = torch.mean(x_extend * y_extend, dim=2)
    return e_xy - e_x_e_y
def metric_fn(preds):
    """
    计算评估指标。

    参数：
    - preds：预测结果DataFrame

    返回：
    - 精确度
    - 召回率
    - 信息系数
    - 排名信息系数
    """
    preds = preds[~np.isnan(preds['label'])]
    precision = {}
    recall = {}
    temp = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by='score', ascending=False))
    if len(temp.index[0]) > 2:
        temp = temp.reset_index(level=0).drop('datetime', axis=1)

    for k in [1, 3, 5, 10, 20, 30, 50, 100]:
        precision[k] = temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / k).mean()
        recall[k] = temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / (x.label > 0).sum()).mean()

    origin_ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score))
    ic = origin_ic.mean()
    icir = origin_ic.mean() / origin_ic.std()
    origin_rank_ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman'))
    rank_ic = origin_rank_ic.mean()
    ricir = rank_ic /origin_rank_ic.std()
    return precision, recall, ic, rank_ic,icir,ricir