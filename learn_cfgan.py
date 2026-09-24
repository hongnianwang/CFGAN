import torch  # 导入 PyTorch 库
import sys
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.optim as optim  # 导入 PyTorch 优化器
import os  # 导入 os 模块，用于操作系统相关功能
import copy  # 导入 copy 模块，用于对象的复制
import json  # 导入 json 模块，用于 JSON 格式数据的处理
import argparse  # 导入 argparse 模块，用于命令行参数解析
import datetime  # 导入 datetime 模块，用于日期时间处理
import collections  # 导入 collections 模块，用于集合数据类型操作
import numpy as np  # 导入 NumPy 库，用于数值计算
import pandas as pd  # 导入 Pandas 库，用于数据处理
from tqdm import tqdm  # 导入 tqdm 模块，用于进度条显示
import qlib  # 导入 qlib 库，用于量化研究
from qlib.config import REG_US, REG_CN  # 导入 qlib 配置信息

from qlib.data.dataset import DatasetH  # 导入 Qlib 数据集模块
from qlib.data.dataset.handler import DataHandlerLP  # 导入 Qlib 数据处理模块
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard 用于可视化
from qlib.contrib.model.pytorch_gru import GRUModel  # 导入 PyTorch GRU 模型
from qlib.contrib.model.pytorch_lstm import LSTMModel  # 导入 PyTorch LSTM 模型
from qlib.contrib.model.pytorch_gats import GATModel  # 导入 PyTorch GAT 模型
from qlib.contrib.model.pytorch_sfm import SFM_Model  # 导入 PyTorch SFM 模型
from qlib.contrib.model.pytorch_alstm import ALSTMModel  # 导入 PyTorch ALSTM 模型
from qlib.contrib.model.pytorch_transformer import Transformer  # 导入 PyTorch Transformer 模型
from cfgan_utils import metric_fn, mse,AverageMeter
from cfgan import CFGAN
from cfgan_dataloader import DataLoader
from qlib.contrib.model.pytorch_tcn import TCNModel
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 根据是否有 GPU 设置设备
# device = 'cpu'
print("device=", device)
EPS = 1e-12  # 定义一个极小值，用于避免除零错误


# 根据模型名称返回对应的模型类
def get_model(model_name):

    if model_name.upper() == 'LSTM':
        return LSTMModel
    if model_name.upper() == 'GRU':
        return GRUModel
    if model_name.upper() == 'GATS':
        return GATModel
    if model_name.upper() == 'SFM':
        return SFM_Model
    if model_name.upper() == 'ALSTM':
        return ALSTMModel
    if model_name.upper() == 'TRANSFORMER':
        return Transformer
    if model_name.upper() =='TCN':
        return TCNModel
    if model_name.upper() =="CFGAN":
        return CFGAN
    raise ValueError('unknown model name `%s`' % model_name)


# 对参数列表进行平均
def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError('the %d-th model has different params' % i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params

# 定义损失函数
def loss_fn(pred, label, args):

    mask = ~torch.isnan(label)  # 过滤掉标签值中的NaN值
    return mse(pred[mask], label[mask])  # 返回均方误差




global_log_file = None  # 全局日志文件路径，初始值为None


def pprint(*args):  # 定义打印函数，可以接收任意数量的参数

    # 打印带有 UTC+8 时间的信息
    time = '[' + str(datetime.datetime.utcnow() +
                     datetime.timedelta(hours=8))[:19] + '] -'  # 获取当前时间并转换为 UTC+8 格式
    print(time, *args, flush=True)  # 打印带有时间的信息，刷新缓冲区

    if global_log_file is None:  # 如果全局日志文件路径为None
        return  # 直接返回，不进行日志记录

    # 将信息写入日志文件
    with open(global_log_file, 'a') as f:  # 打开日志文件，追加模式
        print(time, *args, flush=True, file=f)  # 将带有时间的信息写入日志文件，刷新缓冲区


# 全局变量，用于记录训练步数
global_step = -1


# ==========================
# 训练一个 epoch
# ==========================
def train_epoch(epoch, model, optimizer, train_loader, writer, args, stock2concept_matrix=None):
    global global_step
    model.train()
    # aux_weight = getattr(args, "aux_loss_weight", 0.2)  # 从 args 里读，不存在就默认 0.2

    for i, slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
        global_step += 1
        feature, label, stock_index, _ = train_loader.get(slc)
        feature = torch.nan_to_num(feature, nan=0.0)

        if args.model_name == 'SDGNN':
            out = model(feature, stock_index)
        else:
            out = model(feature)

        if isinstance(out, tuple):
            pred, aux_loss = out
        else:
            pred, aux_loss = out, None

        # 主任务损失
        task_loss = loss_fn(pred, label, args)

        # 总损失
        if aux_loss is not None:
            loss = task_loss + aux_loss
        else:
            loss = task_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()

        # 写入 tensorboard
        writer.add_scalar('Train/Task_Loss', task_loss.item(), global_step)
        if aux_loss is not None:
            writer.add_scalar('Train/Aux_Loss', aux_loss.item(), global_step)
        writer.add_scalar('Train/Total_Loss', loss.item(), global_step)


# ==========================
# 测试一个 epoch
# ==========================
def test_epoch(epoch, model, test_loader, writer, args, stock2concept_matrix=None, prefix='Test'):
    model.eval()

    total_losses, task_losses, aux_losses = [], [], []
    preds = []

    # aux_weight = getattr(args, "aux_loss_weight", 0.2)

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):
        feature, label, stock_index, index = test_loader.get(slc)
        feature = torch.nan_to_num(feature, nan=0.0)

        with torch.no_grad():
            # 模型前向传播 (兼容 tuple / 非 tuple)
            if args.model_name == 'SDGNN':
                out = model(feature, stock_index)
            else:
                out = model(feature)

            if isinstance(out, tuple):
                pred, aux_loss = out
            else:
                pred, aux_loss = out, None

            # 主任务损失
            task_loss = loss_fn(pred, label, args)

            # 总损失
            if aux_loss is not None:
                loss = task_loss + aux_loss
            else:
                loss = task_loss

            preds.append(pd.DataFrame({
                'score': pred.cpu().numpy(),
                'label': label.cpu().numpy()
            }, index=index))

        total_losses.append(loss.item())
        task_losses.append(task_loss.item())
        if aux_loss is not None:
            aux_losses.append(aux_loss.item())

    preds = pd.concat(preds, axis=0)
    precision, recall, ic, rank_ic, icir, ricir = metric_fn(preds)
    scores = ic

    # 写 tensorboard
    writer.add_scalar(prefix + '/Total_Loss', np.mean(total_losses), epoch)
    writer.add_scalar(prefix + '/Task_Loss', np.mean(task_losses), epoch)
    if len(aux_losses) > 0:
        writer.add_scalar(prefix + '/Aux_Loss', np.mean(aux_losses), epoch)

    writer.add_scalar(prefix + '/std(Total_Loss)', np.std(total_losses), epoch)
    writer.add_scalar(prefix + '/' + args.metric, np.mean(scores), epoch)
    writer.add_scalar(prefix + '/std(' + args.metric + ')', np.std(scores), epoch)

    return np.mean(total_losses), scores, precision, recall, ic, rank_ic, icir, ricir

def inference(model, data_loader, args):
    model.eval()
    preds = []

    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):
        feature, label,  stock_index, index = data_loader.get(slc)
        feature = torch.nan_to_num(feature, nan=0.0)

        with torch.no_grad():
    
            if args.model_name == 'SDGNN':
                out = model(feature, stock_index)
            else:
                out = model(feature)

            # 兼容 (pred, aux_loss) 或 pred
            if isinstance(out, tuple):
                pred, _ = out
            else:
                pred = out

            preds.append(pd.DataFrame({
                'score': pred.cpu().numpy(),
                'label': label.cpu().numpy()
            }, index=index))

    preds = pd.concat(preds, axis=0)
    return preds



# 创建数据加载器
def create_loaders(args):
    start_time = datetime.datetime.strptime(args.train_start_date, '%Y-%m-%d')  # 解析训练开始日期
    end_time = datetime.datetime.strptime(args.test_end_date, '%Y-%m-%d')  # 解析测试结束日期
    train_end_time = datetime.datetime.strptime(args.train_end_date, '%Y-%m-%d')  # 解析训练结束日期

    hanlder = {
        'class': 'Alpha360',
        'module_path': 'qlib.contrib.data.handler',
        'kwargs':
            {
                'start_time': start_time,
                'end_time': end_time,
                'fit_start_time': start_time,
                'fit_end_time': train_end_time,
                'instruments': args.data_set,
                'infer_processors': [
                    {
                        'class': 'RobustZScoreNorm',
                        'kwargs':
                            {
                                'fields_group': 'feature',
                                'clip_outlier': True
                            }
                    },
                    {
                        'class': 'Fillna',
                        'kwargs':
                            {
                                'fields_group': 'feature'
                            }
                    }],
                'learn_processors': [
                    {
                        'class': 'DropnaLabel'
                    },
                    {
                        'class': 'CSRankNorm',
                        'kwargs':
                            {
                                'fields_group': 'label'
                            }
                    }],
                'label': ['Ref($close, -1) / $close - 1']
            }
    }
    # 定义数据处理器参数
    segments = {'train': (args.train_start_date, args.train_end_date),
                'valid': (args.valid_start_date, args.valid_end_date),
                'test': (args.test_start_date, args.test_end_date)}  # 定义数据集划分
    # Qlib中创建数据集的方式
    dataset = DatasetH(hanlder, segments)  # 创建数据集

    df_train, df_valid, df_test = dataset.prepare(["train", "valid", "test"], col_set=["feature", "label"],
                                                  data_key=DataHandlerLP.DK_L, )  # 准备数据集
    stock_index = np.load(args.stock_index, allow_pickle=True).item()  # 加载股票索引数据

    start_index = 0  # 设置起始索引

  
    df_train['stock_index'] = 733  # 设置股票索引
    # 处理股票索引，stock_index应该是一个map，stock_index中与instrument索引中匹配的赋值到新列stock_index中，不匹配的默认为733
    df_train['stock_index'] = df_train.index.get_level_values('instrument').map(stock_index).fillna(733).astype(
        int)

    train_loader = DataLoader(df_train["feature"], df_train["label"], df_train['stock_index'],
                              batch_size=args.batch_size, pin_memory=args.pin_memory, start_index=start_index,
                              device=device)  # 创建训练数据加载器

    df_valid['stock_index'] = 733  # 设置股票索引
    df_valid['stock_index'] = df_valid.index.get_level_values('instrument').map(stock_index).fillna(733).astype(
        int)  # 处理股票索引
    start_index += len(df_valid.groupby(level=0).size())  # 更新起始索引

    valid_loader = DataLoader(df_valid["feature"], df_valid["label"], df_valid['stock_index'],
                              pin_memory=True, start_index=start_index, device=device)  # 创建验证数据加载器

    df_test['stock_index'] = 733  # 设置股票索引
    df_test['stock_index'] = df_test.index.get_level_values('instrument').map(stock_index).fillna(733).astype(
        int)  # 处理股票索引
    start_index += len(df_test.groupby(level=0).size())  # 更新起始索引

    test_loader = DataLoader(df_test["feature"], df_test["label"], df_test['stock_index'],
                             pin_memory=True, start_index=start_index, device=device)  # 创建测试数据加载器

    return train_loader, valid_loader, test_loader  # 返回训练、验证和测试数据加载器


# 主函数
def main(args):
    qlib.init(provider_uri=args.provider_uri, region=REG_CN)  # 初始化 Qlib
    print('******************************')

    # seed = np.random.randint(1000000)  # 随机种子
    seed =args.seed
    np.random.seed(seed)  # 设置 NumPy 随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子
    suffix = "%s_dh%s_dn%s_drop%s_lr%s_bs%s_seed%s%s" % (  # 定义模型参数后缀
        args.model_name, args.hidden_size, args.num_layers, args.dropout,
        args.lr, args.batch_size, args.seed, args.annot
    )
    root_path = args.outdir
    print(root_path)

    if not root_path:
        root_path = './output/' + suffix

    os.makedirs(root_path, exist_ok=True)
    os.makedirs(os.path.join(root_path, 'log'), exist_ok=True)
    os.makedirs(os.path.join(root_path, 'pre'), exist_ok=True)
    os.makedirs(os.path.join(root_path, 'mode_save'), exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=root_path)

    # 全局日志文件
    global global_log_file
    global_log_file = os.path.join(
        root_path, 'log', f'{args.name}_run.log'
    )

    # 模型保存目录（单独变量）
    model_save_path = os.path.join(root_path, 'mode_save')
    pprint('create loaders...')  # 输出日志
    pprint("========== Experiment Arguments ==========")
    pprint(vars(args))
    pprint("==========================================")
    train_loader, valid_loader, test_loader = create_loaders(args)  # 创建数据加载器

    stock2concept_matrix = None
    all_precision = []  # 存储所有精度
    all_recall = []  # 存储所有召回率
    all_ic = []  # 存储所有信息系数
    all_rank_ic = []  # 存储所有排序信息系数
    all_icir =[]
    all_ricir =[]
    for times in range(args.repeat):  # 迭代多次
        # seed = np.random.randint(1000000)  # 随机种子
        # # seed =args.seed
        # np.random.seed(seed)  # 设置 NumPy 随机种子
        # torch.manual_seed(seed)  # 设置 PyTorch 随机种子
        pprint(f'seed ={seed} ')  # 输出日志
        pprint('create model...')  # 输出日志
        if args.model_name == 'SFM':  # 如果模型为 SFM
            model = get_model(args.model_name)(d_feat=args.d_feat, output_dim=32, freq_dim=25,
                                               hidden_size=args.hidden_size, dropout_W=0.5, dropout_U=0.5,
                                               device=device)  # 创建 SFM 模型
        elif args.model_name == 'ALSTM':  # 如果模型为 ALSTM
            model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, args.dropout,
                                               'LSTM')  # 创建 ALSTM 模型
        elif args.model_name == 'Transformer':  # 如果模型为 Transformer
            model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers,
                                               dropout=0.5)  # 创建 Transformer 模型
        elif args.model_name == 'CFGAN':
            model = get_model(args.model_name)(d_feat=args.d_feat, seq_len=args.seq_len,hidden_size=args.hidden_size,
                                               num_layers=args.num_layers,num_heads=args.num_heads,
                                               embed_dim=args.embed_dim,dropout= args.dropout)  # 创建 CFGAN 模型
        
        elif args.model_name == 'TCN':
            model =get_model(args.model_name)(num_input=args.d_feat,output_size=1,num_channels=[args.tcn_n_chans] *args.tcn_num_layers,kernel_size=args.tcn_kernel_size,dropout=args.tcn_dropout)
            pass
        else:  # 其他情况
            model = get_model(args.model_name)(d_feat=args.d_feat, num_layers=args.num_layers)  # 创建其他模型

        model.to(device)  # 将模型移到指定设备

        optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=0)  # 使用 Adam 优化器

        
        best_score = -np.inf  # 初始化最佳分数
        best_epoch = 0  # 初始化最佳轮次
        stop_round = 0  # 初始化早停轮数
        best_param = copy.deepcopy(model.state_dict())  # 复制最佳参数
        params_list = collections.deque(maxlen=args.smooth_steps)  # 创建参数列表
        for epoch in range(args.n_epochs):  # 迭代训练轮次
            pprint('Running', times, 'Epoch:', epoch)  # 输出日志

            pprint('training...')  # 输出日志
            train_epoch(epoch, model, optimizer, train_loader, writer, args, stock2concept_matrix)  # 训练模型
            torch.save(model.state_dict(), model_save_path + '/model.bin.e' + str(epoch))
            torch.save(optimizer.state_dict(), model_save_path + '/optimizer.bin.e' + str(epoch))

            params_ckpt = copy.deepcopy(model.state_dict())  # 复制模型参数
            params_list.append(params_ckpt)  # 将模型参数添加到列表
            avg_params = average_params(params_list)  # 计算平均参数
            model.load_state_dict(avg_params)  # 加载平均参数

            pprint('evaluating...')  # 输出日志
            train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic,train_icir,train_ricir = test_epoch(epoch, model,
                                                                                                         train_loader,
                                                                                                         writer, args,
                                                                                                         stock2concept_matrix,
                                                                                                         prefix='Train')  # 计算训练集评价指标
            val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic,val_icir,val_ricir = test_epoch(epoch, model, valid_loader,
                                                                                             writer, args,
                                                                                             stock2concept_matrix,
                                                                                             prefix='Valid')  # 计算验证集评价指标
            test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic,test_icir,test_ricir = test_epoch(epoch, model,
                                                                                                   test_loader, writer,
                                                                                                   args,
                                                                                                   stock2concept_matrix,
                                                                                                   prefix='Test')  # 计算测试集评价指标

            pprint('train_loss %.6f, valid_loss %.6f, test_loss %.6f' % (train_loss, val_loss, test_loss))  # 输出日志
            pprint('train_score %.6f, valid_score %.6f, test_score %.6f' % (train_score, val_score, test_score))  # 输出日志
            pprint('train_ic %.6f, valid_ic %.6f, test_ic %.6f' % (train_ic, val_ic, test_ic))  # 输出日志
            pprint('train_rank_ic %.6f, valid_rank_ic %.6f, test_rank_ic %.6f' % (
                train_rank_ic, val_rank_ic, test_rank_ic))  # 输出日志
            pprint('train_icir %.6f, valid_icir %.6f, test_icir %.6f' % (train_icir, val_icir, test_icir))  # 输出日志
            pprint('train_ricir %.6f, valid_ricir %.6f, test_ricir %.6f' % (train_ricir, val_ricir, test_ricir))  # 输出日志
           
    
            model.load_state_dict(params_ckpt)  # 加载模型参数

            if val_score > best_score:  # 如果验证分数更好
                best_score = val_score  # 更新最佳分数
                best_epoch = epoch  # 更新最佳轮次
                best_param = copy.deepcopy(avg_params)
                stop_round = 0  # 重置早停轮数
            else:  # 否则
                stop_round += 1  # 增加早停轮数
                if stop_round >= args.early_stop:  # 如果早停轮数达到设定值
                    pprint('Early stopping at', epoch, 'with best epoch:', best_epoch)  # 输出日志
                    break  # 退出循环

        model.load_state_dict(best_param)  # 加载最佳参数
        train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic,train_icir,train_ricir  = test_epoch(epoch, model,
                                                                                                     train_loader,
                                                                                                     writer, args,
                                                                                                     stock2concept_matrix,
                                                                                                     prefix='Train')  # 计算训练集评价指标
        val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic,val_icir,val_ricir  = test_epoch(epoch, model, valid_loader,
                                                                                         writer, args,
                                                                                         stock2concept_matrix,
                                                                                         prefix='Valid')  # 计算验证集评价指标
        test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic,test_icir,test_ricir= test_epoch(epoch, model,
                                                                                               test_loader, writer,
                                                                                               args,
                                                                                               stock2concept_matrix,
                                                                                               prefix='Test')  # 计算测试集评价指标
        preds_df = inference(model, test_loader, args)
        pprint('train_ic %.6f, valid_ic %.6f, test_ic %.6f' % (train_ic, val_ic, test_ic))  # 输出日志
        pprint('train_rank_ic %.6f, valid_rank_ic %.6f, test_rank_ic %.6f' % (
        train_rank_ic, val_rank_ic, test_rank_ic))  # 输出日志
        pprint('train_icir %.6f, valid_icir %.6f, test_icir %.6f' % (train_icir, val_icir, test_icir))  # 输出日志
        pprint('train_ricir %.6f, valid_ricir %.6f, test_ricir %.6f' % (train_ricir, val_ricir, test_ricir))  # 输出日志
           
        all_precision.append([train_precision, val_precision, test_precision])  # 存储精度
        all_recall.append([train_recall, val_recall, test_recall])  # 存储召回率
        all_ic.append([train_ic, val_ic, test_ic])  # 存储信息系数
        all_rank_ic.append([train_rank_ic, val_rank_ic, test_rank_ic])  # 存储排序信息系数
        all_icir.append([train_icir,val_icir,test_icir])
        all_ricir.append([train_ricir,val_ricir,test_ricir])
        

    # 汇总统计结果
    # all_precision = np.array(all_precision)  # 转换为 NumPy 数组
    # all_recall = np.array(all_recall)  # 转换为 NumPy 数组
    all_ic = np.array(all_ic)  # 转换为 NumPy 数组
    all_rank_ic = np.array(all_rank_ic)  # 转换为 NumPy 数组
    all_icir = np.array(all_icir)
    all_ricir = np.array(all_ricir)
    # 计算均值和方差
    mean_ic = np.mean(all_ic, axis=0)
    std_ic = np.std(all_ic, axis=0)

    mean_rank_ic = np.mean(all_rank_ic, axis=0)
    std_rank_ic = np.std(all_rank_ic, axis=0)

    mean_icir = np.mean(all_icir, axis=0)
    std_icir = np.std(all_icir, axis=0)

    mean_ricir = np.mean(all_ricir, axis=0)
    std_ricir = np.std(all_ricir, axis=0)

    # 输出均值和方差
    pprint({'all_ic_mean': mean_ic, 'all_ic_std': std_ic})
    pprint({'all_rank_ic_mean': mean_rank_ic, 'all_rank_ic_std': std_rank_ic})
    pprint({'all_icir_ mean': mean_icir, 'all_icir_std': std_icir})
    pprint({'all_ricir_mean': mean_ricir, 'all_ricir_std': std_ricir})
        # 返回验证集 IC 作为 Optuna 优化指标
    return float(mean_ic[1])  # mean_ic[1] -> 验证集 IC
 


class ParseConfigFile(argparse.Action):  # 解析配置文件类

    def __call__(self, parser, namespace, filename, option_string=None):  # 调用函数

        if not os.path.exists(filename):  # 如果文件不存在
            raise ValueError('cannot find config at `%s`' % filename)  # 抛出异常

        with open(filename) as f:  # 打开文件
            config = json.load(f)  # 加载配置
            for key, value in config.items():  # 遍历配置项
                setattr(namespace, key, value)  # 设置属性值

def fmt(date_str):
            # 'YYYY-MM-DD' -> 'YYYYMMDD'
    return date_str.replace('-', '')
def parse_args():  # 解析命令行参数函数

    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--provider_uri', type=str, default='~/.qlib/qlib_data/cn_data')
    # 模型参数
    parser.add_argument('--model_name', default='CFGAN')  # 模型名称，默认为 CFGAN
    parser.add_argument('--d_feat', type=int, default=6)  # 特征维度，默认为6
    parser.add_argument('--hidden_size', type=int, default=96)  # 隐藏层大小，默认为128
    parser.add_argument('--num_layers', type=int, default=2)  # 层数，默认为2
    parser.add_argument('--dropout', type=float, default=0.1)  # Dropout概率，默认为0.0
    parser.add_argument('--K', type=int, default=1)  # K值，默认为1
    parser.add_argument('--seq_len', type=int, default=60,
                        help='input sequence length')
    #CFGAN
    # parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--num_heads', type=int, default=8) 
    parser.add_argument('--embed_dim', type=int, default=96) 
    
      #TCN
    parser.add_argument('--tcn_num_layers', type=int, default=5)  # 层数，默认为5
    parser.add_argument('--tcn_n_chans', type=int, default=128)  # 层数，默认为5
    parser.add_argument('--tcn_kernel_size', type=int, default=3)  # 层数，默认为5
    parser.add_argument('--tcn_dropout', type=float, default=0.5)  # 层数，默认为5
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=200)  # 训练轮次，默认为200
    parser.add_argument('--lr', type=float, default=2e-4)  # 学习率，默认为2e-4
    parser.add_argument('--early_stop', type=int, default=10)  # 早停轮次，默认为30
    parser.add_argument('--smooth_steps', type=int, default=5)  # 平滑步数，默认为5z
    parser.add_argument('--metric', default='IC')  # 评价指标，默vf 认为IC
    # parser.add_argument('--loss', default='mse')  # 损失函数，默认为mse
    
    parser.add_argument('--loss', default='mse', help="选择损失: mse | madl | soft_madl | mse+madl | mse +soft_madl")


    parser.add_argument('--repeat', type=int, default=10)  # 重复次数，默认为10

    # 数据参数
    parser.add_argument('--data_set', type=str, default='csi300')  # 数据集名称，默认为csi300
    parser.add_argument('--pin_memory', action='store_false', default=True)  # 是否将数据存储在固定内存中，默认为True
    parser.add_argument('--batch_size', type=int, default=-1)  # 批大小，默认为-1表示每日批处理
    parser.add_argument('--least_samples_num', type=float, default=1137.0)  # 最小样本数，默认为1137.0
    parser.add_argument('--label', default='')  # 指定其他标签，默认为空
    parser.add_argument('--train_start_date', default='2007-01-01')  # 训练集起始日期，默认为2007-01-01
    parser.add_argument('--train_end_date', default='2017-12-31')  # 训练集结束日期，默认为2014-12-31
    parser.add_argument('--valid_start_date', default='2018-01-01')  # 验证集起始日期，默认为2015-01-01
    parser.add_argument('--valid_end_date', default='2019-12-31')  # 验证集结束日期，默认为2016-12-31
    parser.add_argument('--test_start_date', default='2020-01-01')  # 测试集起始日期，默认为2017-01-01
    parser.add_argument('--test_end_date', default='2023-12-31')  # 测试集结束日期，默认为2020-12-31
    parser.add_argument('--seed', type=int, default=42)  # 随机种子，默认为0
    parser.add_argument('--annot', default='')  # 注释，默认为空
    parser.add_argument('--config', action=ParseConfigFile, default='')  # 配置文件，默认为空
    parser.add_argument('--name', type=str, default='csi300_CFGAN')  # 名称，默认为csi300_CFGAN

    # CSI 300 输入参数
    parser.add_argument('--stock_index',
                        default='./data/csi300_stock_index.npy')  # 股票索引路径，默认为'./data/csi300_stock_index.npy'

    parser.add_argument('--outdir', type=str, default='')  # 输出目录，先设为空
    parser.add_argument('--overwrite', action='store_true', default=False)  # 是否覆盖，默认为False
    args = parser.parse_args()  # 解析参数


    args.outdir = (
        f"./output/"
        f"{args.data_set}/{args.model_name}/"
        f"train_{fmt(args.train_start_date)}_{fmt(args.train_end_date)}/"
        f"valid_{fmt(args.valid_start_date)}_{fmt(args.valid_end_date)}/"
        f"test_{fmt(args.test_start_date)}_{fmt(args.test_end_date)}/"
)
    

    
    
    # print(args.outdir)
    # print(args.outdir )
    return args  # 返回参数

if __name__ == '__main__':  # 如果运行为主程序
    # provider_uri = "~/.qlib/qlib_data/cn_data"  # 设置数据提供路径
    print('******************************')

    args = parse_args()  # 解析命令行参数
    main(args)  # 调用主函数进行训练与评估
