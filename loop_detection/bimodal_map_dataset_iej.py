import numpy as np
import torch
from torch.utils.data import Dataset

class LocusGraphDaset(Dataset):
    def __init__(self, maps, nodes, seqs):
        """
        file_path: 数据集的文件路径
        shape: 数据集的形状，例如 (42832, 256, 40, 40)
        """
        self.maps = maps
        self.seqs = seqs
        self.nodes = nodes
        # self.process()
        print('okk')

    def __len__(self):
        # 数据集中样本的数量
        return len(self.maps)

    def __getitem__(self, idx):
        # 返回单个样本
        return torch.tensor(self.nodes[idx], dtype=torch.float), torch.tensor(self.seqs[idx], dtype=torch.float), torch.tensor(self.maps[idx], dtype=torch.float)

class LocusGraphDasetwoSeq(Dataset):
    def __init__(self, maps, nodes):
        """
        file_path: 数据集的文件路径
        shape: 数据集的形状，例如 (42832, 256, 40, 40)
        """
        self.maps = maps
        # self.nodes = nodes
        self.nodes = self.log_norm(nodes)
        # self.process()
        print('okk')

    def log_norm(self, x):
        return np.log2(x+1)

    def __len__(self):
        # 数据集中样本的数量
        return len(self.maps)

    def __getitem__(self, idx):
        # 返回单个样本
        return torch.tensor(self.nodes[idx], dtype=torch.float), torch.tensor(self.maps[idx], dtype=torch.float)