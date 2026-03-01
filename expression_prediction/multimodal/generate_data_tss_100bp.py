#!/usr/bin/env python
import pathlib
import hicstraw
import argparse
import numpy as np
from dataUtils_tss_100bp import *
from scipy.sparse import save_npz, load_npz
import os
import pickle

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate CAGE Seq expression prediction")
    parser.add_argument('-c', '--cell_line', type=str, default='GM12878')
    parser.add_argument('-b', '--bed', type=str,
                        default='./genelist_filter_GM12878.csv',
                        help='''Path to the bedpe file containing positive training set.''')
    parser.add_argument('-i', '--hic', type=str,
                        default='../../loop_detection/data/csr_matrix/observed_KR',
                        help='''Path to the bedpe file containing positive training set.''')
    parser.add_argument("-d", dest="bigwig_dir", type=str,
                        default='../Input/processed_dir/bigwig/GM12878',
                        help="Path to the chromatin accessibility data which is a bigwig file ")
    parser.add_argument("-o", dest="out_dir", default='./training_data/GM12878_100bp/', help="Folder path to store results.")

    # parser.add_argument("-a", dest="epi", type=list, default=['ATAC', 'DNase', 'CTCF', 'H3K27ac', 'H3K27me3', 'H3K4me3'])
    parser.add_argument("-a", dest="epi", type=list, default=['ATAC', 'DNase'])
    parser.add_argument("-n", dest="norm_type", type=str, default='KR')
    parser.add_argument("-t", dest="data_type", type=str, default='observed')
    parser.add_argument('-l', '--lower', type=int, default=20000,
                        help='''Lower bound of distance between loci in bins (default 2).''')
    parser.add_argument('-u', '--upper', type=int, default=2000000,
                        help='''Upper bound of distance between loci in bins (default 300).''')
    parser.add_argument('-w', '--width', type=int, default=25,
                        help='''Number of bins added to center of window. 
                                default width=11 corresponds to 23*23 windows''')
    parser.add_argument('-r', '--resolution',
                        help='Resolution in bp, default 10000',
                        type=int, default=5000)

    return parser.parse_args()


def main():
    args = get_args()
    np.seterr(divide='ignore', invalid='ignore')

    coords = parsebed(args.bed, lower=args.lower, upper=args.upper, res=args.resolution)#取标记的每对loop的start位置，并在每条染色体上进行排序，生成一个字典包含23条染色体上的所有正样本的两个start位置（除以了10000）
    all_num = sum([len(v) for k,v in coords.items()])
    # with h5py.File(args.label, 'r') as hf:
    #     labels = np.array(hf['targets'])
    # train model per chromosome
    positive_class = {}
    positive_node = {}
    positive_label = {}
    exceptional_key = {
        'GM12878': ['18'],
        'K562': ['3', '4', '9', '15', '18'],
        'HCT116': ['13', '17'],
        'IMR90': ['7'],
        'HepG2': ['10'],
        'WTC11': ['6']
    }
    chromosomes = [f'chr{i}' for i in range(1, 23) if f'{i}' not in exceptional_key[args.cell_line]]

    #
    # labels_dict = {}
    # idx = 0
    # for chr in chromosomes_all:
    #     num = len(coords[chr])
    #     label = labels[idx:idx+num]
    #     labels_dict[chr] = label
    #     idx += num
    coords_new = {}
    for key in chromosomes:
        # if key != '17':
        #     continue
        if key.startswith('chr'):
            chromname = key
        else:
            chromname = 'chr'+key
        file_path = os.path.join(args.hic, args.cell_line, f'{chromname}.npz')
        print('collecting from {}'.format(key))

        if os.path.exists(file_path):
            # 如果文件存在，加载稀疏矩阵
            print(f"loading matrix from {file_path}")
            X = load_npz(file_path)
        else:
            print('fail load matrix')
            exit()

        clist = coords[chromname]
        print('okk')
        clist_tmp = []
        # bigWigFilename = args.bigwig_dir + args.bigwig[0] + '_merged_sorted_RPGC.bigWig'
        for (x, epi_node, epi_edge, label, c) in generateATAC_new(X, clist, chromname, file_dir=args.bigwig_dir, epi_type=args.epi, resou=args.resolution, width=args.width):
            if chromname not in positive_class:
                positive_class[chromname] = []
            if chromname not in positive_node:
                positive_node[chromname] = []
            if chromname not in positive_label:
                positive_label[chromname] = []
            clist_tmp.append(c)

            x = x.astype(np.float32)
            epi_node = epi_node.astype(np.float32)
            label = label.astype(np.float32)

            positive_label[chromname].append(label)
            positive_class[chromname].append(x)
            positive_node[chromname].append(epi_node)
        coords_new[chromname] = clist_tmp
        np.savez_compressed(args.out_dir+'%s_positive.npz' % chromname, data=positive_class[chromname],
                 node=positive_node[chromname], label=positive_label[chromname])

    with open(f'{args.cell_line}_processed_locus.pickle', 'wb') as f:
        pickle.dump(coords_new, f)


if __name__ == "__main__":
    set_seed(42)
    main()
