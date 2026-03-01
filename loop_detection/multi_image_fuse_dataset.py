import numpy as np
import torch
# from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import Dataset

class LocusGraphDaset(Dataset):
    def __init__(self, maps, nodes, seqs, labels):
        """
        file_path: 数据集的文件路径
        shape: 数据集的形状，例如 (42832, 256, 40, 40)
        """
        self.maps = maps
        self.seqs = seqs
        self.labels = labels
        self.nodes = nodes
        # self.process()
        print('okk')

    def __len__(self):
        # 数据集中样本的数量
        return len(self.labels)

    def __getitem__(self, idx):
        # 返回单个样本
        return torch.tensor(self.nodes[idx], dtype=torch.float), torch.tensor(self.seqs[idx], dtype=torch.float), torch.tensor(self.maps[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)

class LocusGraphDasetwoSeq(Dataset):
    def __init__(self, maps, nodes, labels):
        """
        file_path: 数据集的文件路径
        shape: 数据集的形状，例如 (42832, 256, 40, 40)
        """
        self.maps = maps
        self.labels = labels
        # self.nodes = nodes
        self.nodes = self.log_norm(nodes)
        # self.process()
        print('okk')

    def log_norm(self, x):
        return np.log2(x+1)

    def __len__(self):
        # 数据集中样本的数量
        return len(self.labels)

    def __getitem__(self, idx):
        # 返回单个样本
        return torch.tensor(self.nodes[idx], dtype=torch.float), torch.tensor(self.maps[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)

from torchvision import transforms
from PIL import Image
class HiCFoundationDataset(Dataset):
    def __init__(self, maps, labels):
        """
        Args:
            maps (list or np.array): 包含Hi-C图谱的数据集，
                                     每个图谱是一个 50x50 的 numpy 数组。
            labels (list or np.array): 对应的标签。
        """
        self.maps = maps
        self.labels = labels

        # --- 关键修改 ---
        # 1. 定义一个转换流水线，用于调整尺寸和转换为张量
        #    transforms.ToTensor() 会将 PIL Image 或 numpy.ndarray (H x W x C)
        #    转换为 torch.FloatTensor of shape (C x H x W) 并将像素值缩放到 [0.0, 1.0]
        # 2. transforms.Resize() 会将图像调整到目标尺寸。
        #    antialias=True 在进行缩放时可以获得更高质量的结果。
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True)
        ])

        # 3. 定义一个用于标准化的转换。
        #    这是ImageNet预训练模型的标准均值和标准差。
        #    使用这些值可以显著提升模型性能。
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        # --- 修改结束 ---

        print(f"Dataset initialized with {len(self.maps)} samples.")

    def __len__(self):
        # 数据集中样本的数量
        return len(self.labels)

    def __getitem__(self, idx):
        # 获取原始的 50x50 numpy 数组
        hic_map_50x50 = np.squeeze(self.maps[idx])
        pil_image = Image.fromarray(hic_map_50x50)
        map_tensor_1ch = self.transform(pil_image)
        map_tensor_3ch = map_tensor_1ch.repeat(3, 1, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        kk = 0
        return kk, map_tensor_3ch, label

class LocusGraphDatasetWithContactDropout(Dataset):
    def __init__(self, maps, nodes, labels, apply_dropout=False, dropout_rate=0.3):
        """
        参数同上，但这里的dropout_rate代表丢失已有交互的比例。
        """
        self.data_dropout_rate = dropout_rate
        maps = np.squeeze(maps)

        # 高斯噪声
        map_sample_np = self.add_proportional_gaussian_noise(maps, noise_ratio=self.data_dropout_rate)
        # salt
        map_sample_np = self.add_salt_and_pepper_noise(map_sample_np, noise_ratio=self.data_dropout_rate)
        # dropout
        map_sample_np = self.dropdata_inplace(map_sample_np, dropout_rate=self.data_dropout_rate)

        self.maps = map_sample_np
        self.labels = labels
        self.nodes = np.log2(nodes + 1)

        self.apply_dropout = apply_dropout


        print(f"Dropout active: {self.apply_dropout}")

    def __len__(self):
        return len(self.labels)

    def add_salt_and_pepper_noise(self, input_data, noise_ratio=0.05):
        """
        向数据中添加椒盐噪声。

        参数:
        - input_data: 输入的Numpy数组。
        - noise_ratio: 添加噪声的元素比例。

        返回:
        - 添加了噪声的新数组。
        """
        print('Use salt')
        noisy_data = input_data.copy()
        data_min = np.min(input_data)
        data_max = np.max(input_data)

        # 循环处理每个样本
        for i in range(noisy_data.shape[0]):
            sample = noisy_data[i]

            # 计算要添加噪声的像素总数
            num_noise_pixels = int(noise_ratio * sample.size)

            # 添加“盐”噪声 (白色)
            num_salt = num_noise_pixels // 2
            salt_coords = [np.random.randint(0, dim - 1, num_salt) for dim in sample.shape]
            sample[tuple(salt_coords)] = data_max

            # 添加“胡椒”噪声 (黑色)
            num_pepper = num_noise_pixels - num_salt
            pepper_coords = [np.random.randint(0, dim - 1, num_pepper) for dim in sample.shape]
            sample[tuple(pepper_coords)] = data_min

        return noisy_data
    def add_proportional_gaussian_noise(self, input_data, noise_ratio=0.1, std_dev=0.1, mean=0.0):
        """
        向数据中添加可控制比例的高斯噪声。

        参数:
        - input_data: 输入的Numpy数组。
        - noise_ratio: (新增) 要添加噪声的元素的比例，范围 [0, 1]。
        - std_dev: 噪声的标准差，控制噪声的强度。
        - mean: 噪声的均值。

        返回:
        - 添加了噪声的新数组。
        """
        print('Use Gaussian')
        # 复制数据，避免修改原始数组
        noisy_data = input_data.copy()
        data_size = noisy_data.size
        # print('gaussain shape:', noisy_data.shape)
        # 1. 计算要添加噪声的元素数量
        num_noise_elements = int(data_size * noise_ratio)

        # 2. 随机选择要添加噪声的元素索引 (扁平化处理)
        #    np.random.choice 在大数据集上效率很高
        noise_indices = np.random.choice(data_size, num_noise_elements, replace=False)

        # 3. 只为被选中的元素生成高斯噪声
        noise_values = np.random.normal(mean, std_dev, num_noise_elements)

        # 4. 将噪声应用到被选中的位置
        #    np.put 会根据扁平化的索引，将值放入数组中
        np.put(noisy_data, noise_indices, noisy_data.flat[noise_indices] + noise_values)

        # 如果数据有特定范围 (如图像[0,1])，可以取消下面的注释进行裁剪
        # noisy_data = np.clip(noisy_data, data_min, data_max)

        return noisy_data

    def calculate_zero_ratio(self, data):
        """计算矩阵中零元素的比例"""
        if data.size == 0:
            return 1.0
        return np.sum(data == 0) / data.size

    def dropdata_inplace(self, input_data, dropout_rate):
        """
        原地修改数据并返回。
        警告：这个函数会修改你传入的原始 `data` 数组。
        """
        print('use data dropout')
        initial_zero_ratio = self.calculate_zero_ratio(input_data)
        data = input_data.copy()
        # print(data.shape)

        # 循环遍历每个样本
        for i in range(data.shape[0]):

            # 获取当前样本的视图
            sample = data[i]
            # print(sample.shape)

            # 1. 找到矩阵中 *所有* 非零元素的坐标 (行和列)
            #    np.nonzero 是实现这一目标最直接的函数。
            rows, cols = np.nonzero(sample)

            # 如果矩阵已经是全零，就没必要继续了
            if len(rows) == 0:
                continue

            # 2. 计算需要丢弃的元素数量
            num_elements_to_drop = int(len(rows) * dropout_rate)

            if num_elements_to_drop > 0:
                # 3. 从所有非零元素的索引中，随机选择一部分来丢弃
                #    我们随机选择 0 到 len(rows)-1 之间的整数索引。
                indices_to_drop = np.random.choice(
                    len(rows),
                    num_elements_to_drop,
                    replace=False  # 无放回抽样
                )

                # 4. 获取这些被选中要丢弃的元素的实际 (行, 列) 坐标
                drop_rows = rows[indices_to_drop]
                drop_cols = cols[indices_to_drop]

                # 5. 将这些位置的元素置为 0
                #    这是 NumPy 的高级索引功能，可以一次性修改所有选中的点。
                sample[drop_rows, drop_cols] = 0

        final_zero_ratio = self.calculate_zero_ratio(data)

        print(f'Before zero ratio: {initial_zero_ratio:.6f}')
        print(f'After zero ratio:  {final_zero_ratio:.6f}')

        # 因为 data 已经被原地修改，所以直接返回它
        return data

    def __getitem__(self, idx):
        node_sample = torch.tensor(self.nodes[idx], dtype=torch.float)
        label_sample = torch.tensor(self.labels[idx], dtype=torch.float)
        map_sample_np = self.maps[idx].copy()
        # map_sample_np = np.squeeze(map_sample_np)
        #
        map_sample_np = np.expand_dims(map_sample_np, axis=0)
        map_sample = torch.tensor(map_sample_np, dtype=torch.float)

        return node_sample, map_sample, label_sample

class LocusGraphDasetwoSeqFewshot(Dataset):
    def __init__(self, maps, nodes, labels, ratio):
        """
        file_path: 数据集的文件路径
        shape: 数据集的形状，例如 (42832, 256, 40, 40)
        """
        if ratio == 1.0:
            self.maps = maps
            self.labels = labels
            self.nodes = self.log_norm(nodes)
        else:
            self.ori_maps = maps
            self.ori_labels = labels
            self.ori_nodes = nodes

            total_samples = len(self.ori_nodes)
            num_samples = int(np.floor(total_samples * ratio))
            indices = np.random.choice(total_samples, num_samples, replace=False)

            self.maps = self.ori_maps[indices]
            self.labels = self.ori_labels[indices]
            self.nodes = self.log_norm(self.ori_nodes[indices])

        print('okk')

    def log_norm(self, x):
        return np.log2(x+1)

    def __len__(self):
        # 数据集中样本的数量
        return len(self.labels)

    def __getitem__(self, idx):
        # 返回单个样本
        return torch.tensor(self.nodes[idx], dtype=torch.float), torch.tensor(self.maps[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)