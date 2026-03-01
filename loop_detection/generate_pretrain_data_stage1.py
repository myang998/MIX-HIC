#!/usr/bin/env python
import pathlib
import hicstraw
import argparse
import numpy as np
from dataUtils_100bp_bimodal import *
from utils_100bp_bimodal import *
from scipy.sparse import save_npz, load_npz

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
    parser = argparse.ArgumentParser(description="generate positive and negative samples ")

    parser.add_argument("-c", dest="cell_type", default='WTC11', type=str, help="cell line")
    parser.add_argument("-s", dest="signal_type", default='map', type=str, help='[CTCF, map]')
    parser.add_argument('-b', '--bedpe',
                        default='./data/CHIA-PET/GM12878_created_window.txt',
                        help='''Path to the bedpe file containing positive training set.''')

    parser.add_argument("-d", dest="bigwig_dir", type=str,
                        default='../expression_prediction/Input/processed_dir/bigwig/',
                        help="Path to the chromatin accessibility data which is a bigwig file ")

    parser.add_argument("-o", dest="out_dir", default='./training_data', help="Folder path to store results.")
    parser.add_argument("-a", dest="epi", type=list, default=['ATAC', 'DNase'])
    parser.add_argument("-n", dest="norm_type", type=str, default='KR')
    parser.add_argument("-t", dest="data_type", type=str, default='observed')
    # parser.add_argument('-l', '--lower', type=int, default=20000,
    #                     help='''Lower bound of distance between loci in bins (default 2).''')
    parser.add_argument('-l', '--lower', type=int, default=0,
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

    coords = parsebed(args.bedpe, lower=args.lower, upper=args.upper,res=args.resolution)#取标记的每对loop的start位置，并在每条染色体上进行排序，生成一个字典包含23条染色体上的所有正样本的两个start位置（除以了10000）

    # train model per chromosome
    positive_class = {}
    positive_node = {}

    exceptional_key = {
        'GM12878': ['18'],
        'K562': ['3', '4', '9', '15', '18'],
        'HCT116': ['13', '17'],
        'IMR90': ['7'],
        'HepG2': ['10'],
        'WTC11': ['6']
    }

    chromosomes = [str(i) for i in range(1, 23) if str(i) not in exceptional_key[args.cell_type]]

    for key in chromosomes:
        if key.startswith('chr'):
            chromname = key
        else:
            chromname = 'chr'+key
        print('collecting from {}'.format(key))
        file_path = f'./data/csr_matrix/{args.data_type}_{args.norm_type}/{args.cell_type}/{chromname}.npz'
        X = load_npz(file_path)

        clist = coords[chromname]
        print('okk')

        for (x, epi_node, epi_edge) in generateATAC_woseq(X, clist, chromname, file_dir=os.path.join(args.bigwig_dir, args.cell_type), epi_type=args.epi, resou=args.resolution, width=args.width):
            if chromname not in positive_class:
                positive_class[chromname] = []
            if chromname not in positive_node:
                positive_node[chromname] = []

            x = x.astype(np.float32)
            epi_node = epi_node.astype(np.float32)

            positive_class[chromname].append(x)
            positive_node[chromname].append(epi_node)

        print('saving files')
        out_dir = os.path.join(args.out_dir, args.cell_type, f'data_{args.data_type}_{args.norm_type}_{args.signal_type}')
        if not os.path.exists(out_dir):
            print('creating output_dir:', out_dir)
            os.makedirs(out_dir)
        print(chromname, len(positive_class[chromname]))
        np.savez_compressed(out_dir + '/%s.npz' % chromname, data=positive_class[chromname], node=positive_node[chromname])

if __name__ == "__main__":
    set_seed(42)
    main()
