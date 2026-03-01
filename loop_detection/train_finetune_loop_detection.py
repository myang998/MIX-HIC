#!/usr/bin/python

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from collections import OrderedDict
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader

from Models.Bimodal_loop_100bp_v4 import UformerGraphFuse
from trainer_finetune_loop import Trainer

from multi_image_fuse_dataset import LocusGraphDatasetWithContactDropout
from transformers import get_linear_schedule_with_warmup


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Chromatin loops detection for MIX-HIC")
    parser.add_argument("-d", dest="data_dir", type=str, default='./training_data/K562/data_observed_KR_CTCF_100bp',
                        help="A directory containing the training data.")
    parser.add_argument("-l", dest="cell_line", type=str, default='K562',
                        help="Where to save snapshots of the model.")

    # infer_map + pretrain, bimodal + pretrain
    parser.add_argument('--modality', default='bimodal', help='[bimodal, map, epi, infer_map]')
    parser.add_argument('--load_model', default=True, action='store_true')
    parser.add_argument('--pretrain_path', type=str,
                        default='./checkpoints/pretrain/pretrain_dim128.pt',
                        help='path to the saved pre-training model')
    parser.add_argument("-o", dest="loss_ratio", type=float, default=0.0,
                        help="Number of sequences sent to the network in one step.")

    parser.add_argument("-g", dest="gpu", type=str, default='0',
                        help="choose gpu device. eg. '0,1,2' ")
    parser.add_argument("-s", dest="seed", type=int, default=5,
                        help="Random seed to have reproducible results.")
    # Arguments for Adam or SGD optimization
    parser.add_argument("-b", dest="batch_size", type=int, default=128,
                        help="Number of sequences sent to the network in one step.")
    parser.add_argument("-lr", dest="learning_rate", type=float, default=1e-5,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("-e", dest="max_epoch", type=int, default=50,
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
    print('loss ratio: ', args.loss_ratio)
    if torch.cuda.is_available():
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")
    # 使用固定种子，例如：
    set_seed(100)

    # #######在全基因组上######
    exceptional_key = {
        'GM12878': ['18'],
        'K562': ['3', '4', '9', '15', '18'],
        'HCT116': ['13', '17'],
        'IMR90': ['7'],
        'HepG2': ['10'],
        'WTC11': ['6']
    }

    data_pos_files = [f'chr{i}_positive.npz' for i in range(1, 23) if f'{i}' not in exceptional_key[args.cell_line]]
    data_neg_files = [f'chr{i}_negative.npz' for i in range(1, 23) if f'{i}' not in exceptional_key[args.cell_line]]
    data_files = data_pos_files + data_neg_files

    valid_chrom = ['chr10', 'chr11']
    test_chrom = ['chr3', 'chr13', 'chr17']
    train_chrom = [f'chr{i}' for i in range(1, 23) if f'chr{i}' not in valid_chrom + test_chrom]

    train_maps = []
    train_nodes = []
    train_labels = []
    valid_maps = []
    valid_nodes = []
    valid_labels = []
    test_maps = []
    test_nodes = []
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

    print(len(train_maps))
    print(len(valid_maps))
    print(len(test_maps))

    print('3')
    train_maps = np.array(train_maps)
    train_nodes = np.array(train_nodes)
    train_labels = np.array(train_labels)
    # train_seqs = np.array(train_seqs)

    valid_maps = np.array(valid_maps)
    valid_nodes = np.array(valid_nodes)
    valid_labels = np.array(valid_labels)
    # valid_seqs = np.array(valid_seqs)

    test_maps = np.array(test_maps)
    test_nodes = np.array(test_nodes)
    test_labels = np.array(test_labels)
    # test_seqs = np.array(test_seqs)

    train_labels = train_labels.reshape((train_labels.shape[0], 1))
    valid_labels = valid_labels.reshape((valid_labels.shape[0], 1))
    test_labels = test_labels.reshape((test_labels.shape[0], 1))

    train_maps = np.expand_dims(train_maps, axis=1)
    valid_maps = np.expand_dims(valid_maps, axis=1)
    test_maps = np.expand_dims(test_maps, axis=1)
    print('constructing dataset')

    data_dropout_rate = 0.5
    print('data noise rate:', data_dropout_rate)

    train_dataset = LocusGraphDatasetWithContactDropout(train_maps, train_nodes, train_labels, apply_dropout=True, dropout_rate=data_dropout_rate)
    valid_dataset = LocusGraphDatasetWithContactDropout(valid_maps, valid_nodes, valid_labels, apply_dropout=True, dropout_rate=data_dropout_rate)
    test_dataset = LocusGraphDatasetWithContactDropout(test_maps, test_nodes, test_labels, apply_dropout=True, dropout_rate=data_dropout_rate)

    del train_maps, train_nodes, train_labels, valid_maps, valid_nodes, valid_labels, test_maps, test_nodes, test_labels

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    print('Data loading finish')
    print('num of training set:', len(train_dataset))
    print('num of valid set:', len(valid_dataset))
    print('num of test set:', len(test_dataset))

    # implement
    model = UformerGraphFuse(device=device, modality=args.modality, num_heads=args.nheads, embed_dim=args.embed_dim, dropout=args.dropout, depths=args.depths)
    if args.load_model == True:
        print(f'loading models from {args.pretrain_path}')
        # checkpoint = torch.load(args.pretrain_path, map_location='cpu')
        # model.load_state_dict(checkpoint, strict=False)
        #


        checkpoint = torch.load(args.pretrain_path, map_location='cpu')
        new_state_dict = OrderedDict()
        if 'model_state_dict' in checkpoint:
            print('model_state_dict exists')
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:] if k.startswith('module.') else k  # 去掉"module."前缀
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)


        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    else:
        print('Training model from scratch')
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    criterion = nn.BCELoss()
    start_epoch = 0

    # if there exists multiple GPUs, using DataParallel
    if len(args.gpu.split(',')) > 1 and (torch.cuda.device_count() > 1):
        model = nn.DataParallel(model, device_ids=[int(id_) for id_ in args.gpu.split(',')])

    data_type = args.data_dir.split('/')[-1]

    pretrain_ratio = args.pretrain_path.split('/')[-2]
    pretrain_epoch = args.pretrain_path.split('/')[-1].split('_')[1]

    ckpt_file = f'{args.checkpoint}/loop_prediction/{args.cell_line}_{data_type}_{args.modality}_Pretrain{args.load_model}_preepoch{pretrain_epoch}_ratio{pretrain_ratio}_lr{args.learning_rate}_epoch{args.max_epoch}_dim{args.embed_dim}.pth'
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
                       loss_ratio=args.loss_ratio,
                       save_file=ckpt_file,
                       # scheduler=scheduler
                       )

    executor.train()


if __name__ == "__main__":
    main()
