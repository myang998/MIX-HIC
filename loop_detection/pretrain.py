import numpy as np
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import argparse
from scipy.stats import pearsonr,spearmanr
from tqdm import tqdm
from torch.utils.data import Dataset
import os
from Models.LLM_pretrain_100bp_v4 import UformerGraphFuse
import random
from transformers import get_linear_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
import torch
print(torch.cuda.is_available())

def set_seed(seed):
    random.seed(seed)                  # Python随机数生成器
    np.random.seed(seed)               # Numpy随机数生成器
    torch.manual_seed(seed)            # PyTorch CPU种子
    torch.cuda.manual_seed(seed)       # GPU种子（单卡）
    torch.cuda.manual_seed_all(seed)   # GPU种子（多卡）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 确保cudnn一致性

# 使用固定种子，例如：
set_seed(99)

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nheads', default=4, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--ratio', default=1.0, type=float)
    parser.add_argument('--depths', default=2, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--contras_ratio', default=0.5, type=float)

    # parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lr', default=1e-5, type=float)
    # parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--data_dir', type=str, default='./training_data/pretrain', help='path to the load data')

    args = parser.parse_args()
    return args


def prepare_pretrain_data(data_dir):
    """
        dnase: (1, len(chrom))
        ref sequence one hot data: (len(chrom) / binsize, 4, binsize + 600), where binsize = 1000
        cage_data: (13405, 250) , num_seq, seq_len. In sequences.bed file, the length is 250k bp, binsize=1000
    """

    matrix_data = []
    epi_data = []
    chroms = [f'chr{i}' for i in range(1,23)]
    for chr in chroms:
        data = np.load(os.path.join(data_dir, f'{chr}.npz'))
        matrix = data['data']
        epi = data['node']

        matrix_data.extend(matrix)
        epi_data.extend(epi)
    return np.array(matrix_data), np.array(epi_data)


class MyDataset(Dataset):
    def __init__(self, matrix, epi, ratio):
        if ratio == 1.0:
            self.matrix = matrix
            self.epi = epi
        else:
            self.ori_matrix = matrix
            self.ori_epi = epi
            # self.ori_epi = self.log_norm(epi)

            # Calculate the number of samples to select based on the ratio
            total_samples = len(self.ori_epi)
            num_samples = int(np.floor(total_samples * ratio))

            # Randomly select indices according to the ratio
            indices = np.random.choice(total_samples, num_samples, replace=False)

            # Subset the data based on the selected indices
            self.matrix = self.ori_matrix[indices]
            self.epi = self.log_norm(self.ori_epi[indices])

    def log_norm(self, x):
        return np.log2(x+1)

    def __getitem__(self, index):
        return torch.tensor(self.matrix[index], dtype=torch.float), torch.tensor(self.epi[index], dtype=torch.float)

    def __len__(self):
        return len(self.epi)

def get_args():
    args = parser_args()
    return args

def main():

    args = get_args()
    model = UformerGraphFuse(num_heads=args.nheads, embed_dim=args.embed_dim, dropout=args.dropout, depths=args.depths)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-6)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    else:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    args.device = device

    # parser.add_argument('--nheads', default=4, type=int)
    # parser.add_argument('--embed_dim', default=128, type=int)
    # parser.add_argument('--dropout', default=0.0, type=float)
    # parser.add_argument('--ratio', default=1.0, type=float)
    # parser.add_argument('--depths', default=4, type=int)

    print('device being used:', args.device)
    print('dim:', args.embed_dim)
    print('depths:', args.depths)
    print('dropout:', args.dropout)
    print('contras_ratio', args.contras_ratio)
    epi_data = np.memmap(os.path.join(args.data_dir, 'all_node.dat'), dtype=np.float32, mode='r+', shape=(1275948, 5000, 2))
    # epi_data = np.memmap(os.path.join(args.data_dir, 'all_node.dat'), dtype=np.float16, mode='r+', shape=(1275948, 5000, 2))
    print('loading epi finish')

    # from npz
    matrix_data = np.memmap(os.path.join(args.data_dir, 'all_map.dat'), dtype=np.float32, mode='r+', shape=(1275948, 50, 50))

    print('creating dataset')

    pretrain_dataset = MyDataset(matrix_data, epi_data, ratio=args.ratio)
    print(len(matrix_data))
    print(len(pretrain_dataset))
    del epi_data, matrix_data

    if local_rank != -1:
        train_sampler = DistributedSampler(pretrain_dataset, num_replicas=world_size, rank=local_rank)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batchsize, shuffle=(train_sampler is None),
                                      sampler=train_sampler, pin_memory=False, num_workers=8)

    else:
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batchsize, shuffle=True)

    model = UformerGraphFuse(num_heads=args.nheads, embed_dim=args.embed_dim, dropout=args.dropout, depths=args.depths)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-6)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')
    model.to(device)


    # 确保模型参数的存储布局是连续的
    for param in model.parameters():
        param.data = param.data.contiguous()


    start_epoch = 0

    # 加载模型状态
    # checkpoint_path = 'checkpoints/pretrain_41_dim%s_depth%s_ratio%s.pt' % (args.embed_dim, args.depths, args.ratio)
    # start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device=device)

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device), device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )

    for epoch in range(start_epoch, args.epochs):
        print(f'{epoch} start')
        contra_losses = []
        trans_losses = []
        orth_losses = []
        model.train()
        for step, (matrix, epi) in enumerate(tqdm(pretrain_loader)):

            if len(matrix.shape) == 3:
                matrix = matrix.unsqueeze(1)

            # (batch, 1, 50, 50)
            input_matrix = matrix.float().to(device)
            # (batch, 100, 2)
            input_epi = epi.float().to(device)

            (contra_loss, trans_loss, orth_loss) = model(input_matrix, input_epi)

            if epoch < 10:
                total_loss = contra_loss
            else:
                total_loss = args.contras_ratio * + (contra_loss + orth_loss) + (1 - args.contras_ratio) * trans_loss

            # total_loss = contra_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            contra_losses.append(contra_loss.item())
            trans_losses.append(trans_loss.item())
            orth_losses.append(orth_loss.item())

            # 在每个epoch结束时更新学习率
            #scheduler.step((epoch+1) * num_training_batches)
            # scheduler.step()

        # 打印当前学习率
        # print("Epoch:", epoch, "LR:", scheduler.get_last_lr())

        contra_losses = np.average(contra_losses)
        trans_losses = np.average(trans_losses)
        orth_losses = np.average(orth_losses)

        if local_rank in [-1, 0]:
            print('Epoch: {} all_loss: {:.7f}, contra_losses: {:.7f}, trans_losses: {:.7f}, orth_losses: {:.7f}'.format(epoch, contra_losses + trans_losses + orth_losses, contra_losses, trans_losses, orth_losses))

            if (epoch + 1) % 1 == 0:
                print(f'save model at {epoch}')
                # torch.save(model.state_dict(), 'checkpoints/pretrain_%s_dim%s_ratio%s.pt' % (epoch, args.embed_dim, args.ratio))
                # save_checkpoint(model, optimizer, epoch, 'checkpoints/pretrain_%s_dim%s_depth%s_ratio%s.pt' % (epoch, args.embed_dim, args.depths, args.ratio))

    # torch.save(model.state_dict(), 'checkpoints/pretrain_final_dim%s_ratio%s.pt' % (args.embed_dim, args.ratio))

# 保存 checkpoint 函数
def save_checkpoint(model, optimizer, epoch, checkpoint_path, scheduler=None):
    checkpoint = {
        'epoch': epoch + 1,  # 保存当前epoch
        'model_state_dict': model.state_dict(),  # 模型参数
        'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
        # 'scheduler_state_dict': scheduler.state_dict(),  # 学习率调度器状态
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1}")

# 加载 checkpoint 函数
def load_checkpoint(model, optimizer, checkpoint_path, device, scheduler=None):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] if k.startswith('module.') else k  # 去掉"module."前缀
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from epoch {epoch}")
        return epoch
    else:
        print("No checkpoint found, starting from scratch.")
        return 0


if __name__=="__main__":
    main()
