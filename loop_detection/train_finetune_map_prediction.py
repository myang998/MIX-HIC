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

from Models.Bimodal_map_100bp_v4 import UformerGraphFuse
from trainer_finetune_map import Trainer

from bimodal_map_dataset_iej import LocusGraphDasetwoSeq


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description=" DLoopCaller train model for chromatin loops")
    parser.add_argument("-d", dest="data_dir", type=str, default='../expression_prediction/multimodal/training_data/GM12878_100bp/',
                        help="A directory containing the training data.")

    parser.add_argument("-l", dest="cell_line", type=str, default='GM12878',
                        help="Where to save snapshots of the model.")


    parser.add_argument('--modality', default='infer_map', help='[epi, infer_map]')
    parser.add_argument('--load_model', default=True, action='store_true')
    parser.add_argument('--pretrain_path', type=str,
                        default='./checkpoints/pretrain/pretrain_dim128.pt',
                        help='path to the saved pre-training model')
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("-g", dest="gpu", type=str, default='0',
                        help="choose gpu device. eg. '0,1,2' ")
    parser.add_argument("-s", dest="seed", type=int, default=5,
                        help="Random seed to have reproducible results.")
    # Arguments for Adam or SGD optimization
    parser.add_argument("-b", dest="batch_size", type=int, default=32,
                        help="Number of sequences sent to the network in one step.")
    parser.add_argument("-lr", dest="learning_rate", type=float, default=1e-5,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("-e", dest="max_epoch", type=int, default=200,
                        help="Number of training steps.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./checkpoints/',
                        help="Where to save snapshots of the model.")

    parser.add_argument('--nheads', default=4, type=int)
    parser.add_argument('--embed_dim', default=128, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--depths', default=2, type=int)

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
    print('dim', args.embed_dim)
    print('lr',args.learning_rate)
    print('dropout', args.dropout)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")
    set_seed(100)
    exceptional_key = {
        'GM12878': ['18'],
        'K562': ['3', '4', '9', '15', '18'],
        'HCT116': ['13', '17'],
        'IMR90': ['7'],
        'HepG2': ['10'],
        'WTC11': ['6']
    }

    # #######在全基因组上######
    data_pos_files = [f'chr{i}_positive.npz' for i in range(1, 23) if f'{i}' not in exceptional_key[args.cell_line]]
    edge_pos_files = [f'chr{i}_positive_edge.pkl' for i in range(1, 23) if f'{i}' not in exceptional_key[args.cell_line]]
    # seq_pos_files = [f'chr{i}_positive_seq.pkl' for i in range(1, 23) if f'{i}' not in exceptional_key[args.cell_line]]

    data_files = data_pos_files
    edge_files = edge_pos_files
    # seq_files = seq_pos_files

    valid_chrom = ['chr10', 'chr11']
    test_chrom = ['chr3', 'chr13', 'chr17']
    train_chrom = [f'chr{i}' for i in range(1, 23) if f'chr{i}' not in valid_chrom + test_chrom]

    train_maps = []
    train_nodes = []
    # train_seqs = []
    train_labels = []
    valid_maps = []
    valid_nodes = []
    # valid_seqs = []
    valid_labels = []
    test_maps = []
    test_nodes = []
    # test_seqs = []
    test_labels = []

    print('2')
    matrix_data = []
    epi_data = []
    cage_data = []
    for filename in data_files:
        Data = np.load(os.path.join(args.data_dir, filename))

        data = Data['data']
        nodes = Data['node']
        label = Data['label']

        # 分不同chromosomes
        if filename.split('_')[0] in train_chrom:
            train_maps.extend(data)
            train_nodes.extend(nodes)
            train_labels.extend(label)
        elif filename.split('_')[0] in valid_chrom:
            valid_maps.extend(data)
            valid_nodes.extend(nodes)
            valid_labels.extend(label)
        else:
            test_maps.extend(data)
            test_nodes.extend(nodes)
            test_labels.extend(label)

    print('3')
    train_maps = np.array(train_maps)
    train_nodes = np.array(train_nodes)
    # train_seqs = np.array(train_seqs)

    valid_maps = np.array(valid_maps)
    valid_nodes = np.array(valid_nodes)
    # valid_seqs = np.array(valid_seqs)

    test_maps = np.array(test_maps)
    test_nodes = np.array(test_nodes)

    train_maps = np.expand_dims(train_maps, axis=1)
    valid_maps = np.expand_dims(valid_maps, axis=1)
    test_maps = np.expand_dims(test_maps, axis=1)
    print('constructing dataset')

    train_dataset = LocusGraphDasetwoSeq(train_maps, train_nodes)
    valid_dataset = LocusGraphDasetwoSeq(valid_maps, valid_nodes)
    test_dataset = LocusGraphDasetwoSeq(test_maps, test_nodes)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    print('Data loading finish')
    print('num of training set:', len(train_dataset))
    print('num of valid set:', len(valid_dataset))
    print('num of test set:', len(test_dataset))

    # implement
    model = UformerGraphFuse(modality=args.modality, num_heads=args.nheads, embed_dim=args.embed_dim, dropout=args.dropout, depths=args.depths)

    if args.load_model == True:
        print(f'loading models from {args.pretrain_path}')

        checkpoint = torch.load(args.pretrain_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            print('load from dict')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    else:
        print('Training model from scratch')
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    criterion = nn.MSELoss()
    start_epoch = 0

    # if there exists multiple GPUs, using DataParallel
    if len(args.gpu.split(',')) > 1 and (torch.cuda.device_count() > 1):
        model = nn.DataParallel(model, device_ids=[int(id_) for id_ in args.gpu.split(',')])

    data_type = args.data_dir.split('/')[-1]
    pretrain_epoch = args.pretrain_path.split('/')[-1].split('_')[1]
    pretrain_ratio = args.pretrain_path.split('/')[-2]
    ckpt_file = f'{args.checkpoint}/map_prediction_iej/{args.cell_line}_{args.modality}_Pretrain{args.load_model}_ratio{pretrain_ratio}_pretrainepoch{pretrain_epoch}_lr{args.learning_rate}_epoch{args.max_epoch}_dim{args.embed_dim}.pth'
    early_stopping_patience = args.early_stopping_patience
    executor = Trainer(model=model,
                       optimizer=optimizer,
                       criterion=criterion,
                       device=device,
                       checkpoint=args.checkpoint,
                       start_epoch=start_epoch,
                       max_epoch=args.max_epoch,
                       train_loader=train_loader,
                       valid_loader=valid_loader,
                       test_loader=test_loader,
                       lr_policy=None,
                       save_file=ckpt_file,
                       early_stopping_patience=early_stopping_patience)

    executor.train()


if __name__ == "__main__":
    main()
