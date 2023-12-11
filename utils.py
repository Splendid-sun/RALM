import sys
import gc
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import random
import time

def get_now_time_fullstring():
    # 返回当前时间的完整字符串格式，例如 "2023-06-05 16:35:00"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def format_instruction(row, mode):
    # 格式化数据行，将其转换为文本格式
    # row 是一个字典，包含了一系列的问题和答案
    # mode 是一个字符串，表示当前的模式（训练或者测试）
    
    text = ""
    text += "The following are multiple choice questions (with answers) that include context\n"
    text += "\n"
    text += "Context:\n"
    for i in range(5):
        text += f"{row['context'][i]}\n"
        if i != 4:
            text += "###\n"
    text += "\n"
    text += "Question:\n"
    text += f"{row['prompt']}\n"
    text += "\n"
    text += "Options:\n"
    text += f"A: {row['A']}\n"
    text += f"B: {row['B']}\n"
    text += f"C: {row['C']}\n"
    text += f"D: {row['D']}\n"
    text += f"E: {row['E']}\n"
    text += "\n"
    text += "Answer: "

    if mode == "train":
        text += f"{row['answer']}"
    return text

def plot_sequence_lengths(data, split='train', max_length=2048, is_plot=False):
    # 计算并绘制文本序列长度的分布
    # data 是一个字典，包含了训练和测试数据
    # split 是一个字符串，表示当前的数据切分（训练或者测试）
    # max_length 是一个整数，表示序列的最大长度
    # is_plot 是一个布尔值，表示是否需要绘制直方图
    
    sequence_lengths = []
    keep_indices = []

    # 遍历数据集，获取文本序列的长度
    for i, example in enumerate(data[split]):
        sequence_lengths.append(len(example['instruction']))
        if sequence_lengths[i] < max_length:
            keep_indices.append(i)
    
    if is_plot:
        # 绘制直方图
        plt.hist(sequence_lengths, bins=30)
        plt.xlabel('Sequence Length')
        plt.ylabel('Count')
        plt.title('Distribution of Text Sequence Lengths')
        plt.show()

    return keep_indices

def seed_everything(seed=42):
    # 设置随机种子，以确保实验的可重复性
    # seed 是一个整数，表示随机种子
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 

def init_logger(log_file):
    # 初始化日志器
    # log_file 是一个字符串，表示日志文件的路径
    
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def get_timediff(time1,time2):
    # 计算两个时间点之间的时间差，并以分钟和秒的形式返回
    # time1 和 time2 是两个时间点，以秒为单位
    
    minute_,second_ = divmod(time2-time1,60)
    return f"{int(minute_):02d}:{int(second_):02d}"
