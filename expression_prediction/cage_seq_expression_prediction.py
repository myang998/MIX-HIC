from cage.util import prepare_train_data_multimodal

import os, pickle, time
import random
import numpy as np
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import argparse
from scipy.stats import pearsonr,spearmanr
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.metrics import r2_score
# from multimodal.Models.Uencoder_bimodal import UformerGraphFuse
from multimodal.Models.Bimodal_GEP_v4_100bp import UformerGraphFuse
# from Models.Bimodal_GEP_v1 import UformerGraphFuse
from collections import OrderedDict

def set_seed(seed):
    random.seed(seed)                  # Python随机数生成器
    np.random.seed(seed)               # Numpy随机数生成器
    torch.manual_seed(seed)            # PyTorch CPU种子
    torch.cuda.manual_seed(seed)       # GPU种子（单卡）
    torch.cuda.manual_seed_all(seed)   # GPU种子（多卡）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 确保cudnn一致性

# 使用固定种子，例如：
set_seed(1200)


def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--loss_rate', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--downsample_ratio', type=float, default=0)
    parser.add_argument('--use_rope', default=False)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--depths', type=int, default=2)

    # bimodal + pretrain, infer_map + pretrain
    parser.add_argument('--modality', default='bimodal', help='[bimodal, epi, infer_map]')
    parser.add_argument('--load_model', default=False, action='store_true')
    parser.add_argument('--device', default='0')
    parser.add_argument('--pretrain_path', type=str, default='../loop_detection/checkpoints/pretrain/pretrain_dim256.pt',help='path to the saved pre-training model')

    parser.add_argument('--data_dir', type=str, default='./multimodal/training_data/GM12878_100bp', help='path to the load data')
    parser.add_argument('--cell_type', type=str, default='GM12878')
    args = parser.parse_args()
    return args

class MyDataset(Dataset):
    def __init__(self, locus, matrix, epi, label):

        self.matrix = matrix[locus]
        # self.epi = epi[locus]
        self.epi = self.log_norm(epi[locus])
        self.label = self.log_norm(label[locus])
        self.label = np.tile(self.label, (1, 2))
        print('ok')

    def log_norm(self, x):
        return np.log2(x+1)

    def __getitem__(self, index):
        return torch.tensor(self.matrix[index], dtype=torch.float), torch.tensor(self.epi[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.float)
    def __len__(self):
        return len(self.label)

def get_args():
    args = parser_args()
    return args

def shuffle_data(dataset_size,seed=8):
    # randomly split training/validation/testing sets
    indices=np.arange(dataset_size)
    valid_split=int(np.floor(dataset_size*0.8))
    test_split=int(np.floor(dataset_size*0.9))
    np.random.seed(seed)
    np.random.shuffle(indices)
    return indices[:valid_split],indices[valid_split:test_split],indices[test_split:]

def main():

    args = get_args()
    model= UformerGraphFuse(embed_dim=args.embed_dim, modality=args.modality, use_rope=args.use_rope, num_heads=args.num_heads, depths=args.depths, dropout=args.dropout)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print('device being used:', device)
    print('repo positing:', args.use_rope)
    print('cell type:', args.cell_type)
    print('lr:', args.lr)
    # load model
    if args.load_model == True:
        print(f'loading models from {args.pretrain_path}')
        # model_dict = model.state_dict()
        # pretrain_dict = torch.load(args.pretrain_path, map_location='cpu')
        # pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        #
        # model_dict.update(pretrain_dict)
        # model.load_state_dict(model_dict)

        # 如果有state_dict以及optimizer的时候就需要这样加载
        checkpoint = torch.load(args.pretrain_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] if k.startswith('module.') else k  # 去掉"module."前缀
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)

    model.to(f'cuda:{args.device}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    criterion= torch.nn.MSELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,weight_decay=1e-6)

    exceptional_key = {
        'GM12878': ['chr18'],
        'K562': ['chr3', 'chr4', 'chr9', 'chr15', 'chr18'],
        'HCT116': ['chr13', 'chr17'],
        'IMR90': ['chr7'],
        'HepG2': ['chr10'],
        'WTC11': ['chr6']
    }

    matrix_data, epi_data, cage_data = prepare_train_data_multimodal(args.cell_type, args.data_dir, exceptional_key[args.cell_type])
    print('all data num: ', len(matrix_data))
    if args.downsample_ratio != 0:
        total_samples = len(matrix_data)
        num_samples = int(np.floor(total_samples * args.downsample_ratio))

        indices = np.random.choice(total_samples, num_samples, replace=False)

        matrix_data = matrix_data[indices]
        epi_data = epi_data[indices]
        cage_data = cage_data[indices]

    train_index, valid_index, test_index = shuffle_data(dataset_size=len(matrix_data))

    train_dataset = MyDataset(train_index, matrix_data, epi_data, cage_data)
    valid_dataset = MyDataset(valid_index, matrix_data, epi_data, cage_data)
    test_dataset = MyDataset(test_index, matrix_data, epi_data, cage_data)

    train_loader=DataLoader(train_dataset,batch_size=args.batchsize,shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=True)
    pretrain_epoch = args.pretrain_path.split('/')[-1].split('_')[1]
    ckpt_file = f'./checkpoints/{args.cell_type}_{args.modality}_Pretrain{args.load_model}_pretrainepoch{pretrain_epoch}_lr{args.lr}_epoch{args.epochs}_batch{args.batchsize}_repo{args.use_rope}_dim{args.embed_dim}_depth{args.depths}_head{args.num_heads}_dropout{args.dropout}.pth'

    early_stopping_patience = 20
    no_improvement_count = 0

    best_criter=0
    for epoch in range(args.epochs):
        training_losses=[]
        model.train()
        pred_eval = []
        target_eval = []
        for step, (matrix, epi, input_label) in enumerate(tqdm(train_loader)):
            # if step ==3:
            #     break
            if len(matrix.shape) == 3:
                matrix = matrix.unsqueeze(1)
            t=time.time()

            # (batch, 1, 50, 50)
            input_matrix = matrix.float().to(device)
            # (batch, 100, 2)
            input_epi = epi.float().to(device)

            # (batch, 100)
            input_label = input_label.float().to(device)

            output, contras_loss = model(input_matrix, input_epi)
            loss = criterion(output, input_label)

            loss = loss / args.accum_iter
            total_loss = loss + args.loss_rate * contras_loss
            total_loss.backward()
            if ((step + 1) % args.accum_iter == 0) or (step + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            cur_loss = loss.item()
            training_losses.append(cur_loss)

            pred_eval.append(output.cpu().data.detach().numpy())
            target_eval.append(input_label.cpu().data.detach().numpy())

        pred_eval = np.concatenate(pred_eval, axis=0).squeeze().flatten()
        target_eval = np.concatenate(target_eval, axis=0).squeeze().flatten()

        epoch_r2 = r2_score(pred_eval, target_eval)
        epoch_pearsonr, _ = pearsonr(pred_eval, target_eval)
        epoch_spearmanr, _ = spearmanr(pred_eval, target_eval)

        train_loss = np.average(training_losses)
        print('Epoch: {} LR: {:.8f} train_loss: {:.7f}, R2: {:.5f}, PCC: {:.5f}, SRCC: {:.5f}'.format(epoch, optimizer.param_groups[0]['lr'], train_loss, epoch_r2, epoch_pearsonr, epoch_spearmanr))
        model.eval()

        validation_losses = []
        pred_eval = []
        target_eval = []

        for step, (matrix, epi, input_label) in enumerate(valid_loader):
            with torch.no_grad():
                if len(matrix.shape) == 3:
                    matrix = matrix.unsqueeze(1)
                input_label = input_label.float().to(device)
                input_matrix = matrix.float().to(device)
                input_epi = epi.float().to(device)
                input_label = input_label.float().to(device)

                output, contras_loss = model(input_matrix, input_epi)
                loss = criterion(output, input_label)

                cur_loss = loss.item()
                if isinstance(contras_loss, torch.Tensor):
                    contras_loss = contras_loss.cpu()
                validation_losses.append(cur_loss + args.loss_rate * contras_loss)

                pred_eval.append(output.cpu().data.detach().numpy())
                target_eval.append(input_label.cpu().data.detach().numpy())
        pred_eval = np.concatenate(pred_eval, axis=0).squeeze().flatten()
        target_eval = np.concatenate(target_eval, axis=0).squeeze().flatten()

        epoch_r2 = r2_score(pred_eval, target_eval)
        epoch_pearsonr, _ = pearsonr(pred_eval, target_eval)
        epoch_spearmanr, _ = spearmanr(pred_eval, target_eval)
        validation_losses = [x.numpy() for x in validation_losses]
        validation_losses = np.array(validation_losses)  # 或者直接使用原列表

        # print(validation_losses)

        print('Validation epoch: {} LR: {:.8f} train_loss: {:.7f}, R2: {:.5f}, PCC: {:.5f}, SRCC: {:.5f}'.format(epoch,
                                                                                                      optimizer.param_groups[
                                                                                                          0]['lr'],
                                                                                                      np.average(validation_losses),
                                                                                                      epoch_r2,
                                                                                                      epoch_pearsonr,
                                                                                                      epoch_spearmanr))


        if epoch_pearsonr > best_criter:
            no_improvement_count = 0
            best_criter = epoch_pearsonr
            print('save model to:', ckpt_file)
            # torch.save(model.state_dict(), ckpt_file)

            test_losses = []
            pred_eval = []
            target_eval = []

            for step, (matrix, epi, input_label) in enumerate(test_loader):
                with torch.no_grad():
                    if len(matrix.shape) == 3:
                        matrix = matrix.unsqueeze(1)
                    input_label = input_label.float().to(device)
                    input_matrix = matrix.float().to(device)
                    input_epi = epi.float().to(device)
                    input_label = input_label.float().to(device)

                    output, contras_loss = model(input_matrix, input_epi)
                    loss = criterion(output, input_label)

                    cur_loss = loss.item()
                    if isinstance(contras_loss, torch.Tensor):
                        contras_loss = contras_loss.cpu()
                    test_losses.append(cur_loss + args.loss_rate * contras_loss)
                    pred_eval.append(output.cpu().data.detach().numpy())
                    target_eval.append(input_label.cpu().data.detach().numpy())
            pred_eval = np.concatenate(pred_eval, axis=0).squeeze().flatten()
            target_eval = np.concatenate(target_eval, axis=0).squeeze().flatten()

            epoch_r2 = r2_score(pred_eval, target_eval)
            epoch_pearsonr, _ = pearsonr(pred_eval, target_eval)
            epoch_spearmanr, _ = spearmanr(pred_eval, target_eval)
            test_losses = [x.numpy() for x in test_losses]
            test_losses = np.array(test_losses)
            print('test epoch: {} LR: {:.8f} train_loss: {:.7f}, R2: {:.5f}, PCC: {:.5f}, SRCC: {:.5f}'.format(
                epoch,
                optimizer.param_groups[
                    0]['lr'],
                np.average(test_losses),
                epoch_r2,
                epoch_pearsonr,
                epoch_spearmanr))
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_patience:
            print(f'No improvement in validation loss for {early_stopping_patience} epochs. Early stopping!')
            break

if __name__=="__main__":
    main()