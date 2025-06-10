import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import datasets 

#! 直接从处理好的数据集中读取
class PretrainDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.dataset = self.load_data(data_path)
        print(f"数据路径: {data_path}")
        print(f"文件列表: {os.listdir(data_path)}")

    def load_data(self, data_path):
        # 直接从预处理好的数据集加载
        try:
            dataset = datasets.load_from_disk(data_path)
            print(f"成功加载预处理数据集，包含 {len(dataset)} 个样本")
            return dataset
        except Exception as e:
            raise ValueError(f"无法从 {data_path} 加载预处理数据集: {e}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        input_ids = torch.tensor(sample['input_ids'], dtype=torch.long)
        
        # 确保 loss_mask 是张量
        if isinstance(sample['loss_mask'], torch.Tensor):
            loss_mask = sample['loss_mask']
        else:
            loss_mask = torch.tensor(sample['loss_mask'], dtype=torch.long)
        
        # 构建训练数据
        X = input_ids[:-1]  # 去掉最后一个 token 作为输入
        Y = input_ids[1:]   # 去掉第一个 token 作为目标
        loss_mask = loss_mask[1:]  # 对齐预测位置
        
        return X, Y, loss_mask

