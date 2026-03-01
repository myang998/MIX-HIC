#!/usr/bin/python

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from tqdm import tqdm
from Models.Bimodal_loop_100bp_v4 import UformerGraphFuse

from multi_image_fuse_dataset import LocusGraphDasetwoSeq
from sklearn.metrics import roc_auc_score, precision_score,f1_score,recall_score,average_precision_score

def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Independent test for chromatin loops detection")
    parser.add_argument("-d", dest="data_dir", type=str, default='./training_data/K562/CTCF_dataset', help="A directory containing the training data.")
    parser.add_argument("-l", dest="cell_line", type=str, default='K562',
                        help="Where to save snapshots of the model.")

    # epi + pretrain, epi + wo/pretrain, infer_map + pretrain
    parser.add_argument('--modality', default='infer_map', help='[bimodal, map, epi, infer_map]')
    parser.add_argument('--load_model', default=True, action='store_true')
    parser.add_argument('--pretrain_path', type=str, default='./checkpoints/loop_prediction/GM12878_data_observed_KR_CTCF_100bp_infer_map_PretrainTrue_lr1e-05_epoch50_dim128.pth',help='path to the saved pre-training model')

    parser.add_argument("-g", dest="gpu", type=str, default='0',
                        help="choose gpu device. eg. '0,1,2' ")
    parser.add_argument("-s", dest="seed", type=int, default=5,
                        help="Random seed to have reproducible results.")
    # Arguments for Adam or SGD optimization
    parser.add_argument("-b", dest="batch_size", type=int, default=128,
                        help="Number of sequences sent to the network in one step.")
    parser.add_argument("-lr", dest="learning_rate", type=float, default=1e-5,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("-e", dest="max_epoch", type=int, default=100,
                        help="Number of training steps.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./checkpoints/',
                        help="Where to save snapshots of the model.")

    parser.add_argument('--nheads', default=4, type=int)
    parser.add_argument('--embed_dim', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--depths', default=2, type=int)

    parser.add_argument('--mode', default='ground_truth', type=float, choice="pesudo, ground_truth")
    parser.add_argument('--remove_ratio', default=0.0, type=float)
    parser.add_argument('--way', default='reduce', type=float, choice="reduce, remove")


    return parser.parse_args()

def shuffle_data(dataset_size,seed=8):
    # randomly split training/validation/testing sets
    indices=np.arange(dataset_size)
    valid_split=int(np.floor(dataset_size*0.8))
    test_split=int(np.floor(dataset_size*0.9))
    np.random.seed(seed)
    np.random.shuffle(indices)
    return indices[:valid_split],indices[valid_split:test_split],indices[test_split:]

def set_seed(seed):
    random.seed(seed)                  # Python随机数生成器
    np.random.seed(seed)               # Numpy随机数生成器
    torch.manual_seed(seed)            # PyTorch CPU种子
    torch.cuda.manual_seed(seed)       # GPU种子（单卡）
    torch.cuda.manual_seed_all(seed)   # GPU种子（多卡）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 确保cudnn一致性

def main():
    """Create the model and start the training."""
    args = get_args()
    print(args.modality)
    print('load model:', args.load_model)
    print(args.data_dir)
    print('gpu:', args.gpu)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")
    set_seed(100)

    test_chrom = ['chr1', 'chr2', 'chr5', 'chr6', 'chr7', 'chr8', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr16', 'chr17', 'chr19', 'chr20', 'chr22']

    if args.mode != 'pesudo':
        data_pos_files = [f'{i}_positive_{args.mode}_{args.remove_ratio}_{args.way}.npz' for i in test_chrom]
    else:
        data_pos_files = [f'{i}_positive.npz' for i in test_chrom]
    data_files = data_pos_files

    print('2')
    matrix_data = []
    epi_data = []
    cage_data = []
    for filename in data_files:
        print(filename)
        Data = np.load(os.path.join(args.data_dir, filename))

        data = Data['data']
        nodes = Data['node']
        label = Data['label']

        # 不分chromosomes
        matrix_data.extend(data)
        epi_data.extend(nodes)
        cage_data.extend(label)


    # 全部chromosomes
    matrix_data, epi_data, cage_data = np.array(matrix_data), np.array(epi_data), np.array(cage_data)
    matrix_data = np.expand_dims(matrix_data, axis=1)

    print('num of test set:', len(matrix_data))

    # test set不分chromosomes
    test_chrom_seqs = np.array(matrix_data)
    test_chrom_epis = np.array(epi_data)
    test_chrom_labels = np.array(cage_data)
    print('3')

    # implement
    model = UformerGraphFuse(modality=args.modality, num_heads=args.nheads, embed_dim=args.embed_dim, dropout=args.dropout, depths=args.depths).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    if args.load_model == True:
        print(f'loading models from {args.pretrain_path}')
        checkpoint = torch.load(args.pretrain_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
    model.eval()

    criterion = nn.BCELoss()


    print('test chromosomes no split dataset')

    test_dataset = LocusGraphDasetwoSeq(test_chrom_seqs, test_chrom_epis, test_chrom_labels)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print('num of test set:', len(test_dataset))

    label_p_all = []
    label_t_all = []
    for i_batch, sample_batch in enumerate(tqdm(test_loader)):
        data_graph = sample_batch[0].float().to(device)
        data_map = sample_batch[1].float().to(device)
        label = sample_batch[2].float().to(device)
        with torch.no_grad():
            label_p, _ = model(data_graph, data_map)
        # print(label_p)
        label_p_all.extend(label_p.view(-1).data.cpu().numpy())
        label_t_all.extend(label.view(-1).data.cpu().numpy())
    loss = criterion(torch.tensor(label_p_all), torch.tensor(label_t_all))
    r = recall_score(label_t_all, [int(x > 0.5) for x in label_p_all])
    p = precision_score(label_t_all, [int(x > 0.5) for x in label_p_all])
    f1 = f1_score(label_t_all, [int(x > 0.5) for x in label_p_all])
    valid_auc = roc_auc_score(label_t_all, label_p_all)
    valid_aupr = average_precision_score(label_t_all, label_p_all)
    print("Test loss:{:.6f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, auc:{:.4f}, aupr:{:.4f}".format(loss.item(), p, r, f1,
                                                                                               valid_auc, valid_aupr))

if __name__ == "__main__":
    main()
