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

    parser.add_argument("-c", dest="cell_type", default='GM12878', type=str, help="cell line")
    parser.add_argument('-b', '--bedpe',
                        default='./data/CHIA-PET/GM12878_CTCF_ENCSR184YZV.txt',
                        help='''Path to the loop bedpe file containing positive training set.''')
    parser.add_argument("-o", dest="out_dir", default='./training_data/GM12878/data_observed_KR_CTCF_100bp/', help="Folder path to store results.")
    parser.add_argument("-d", dest="bigwig_dir", type=str,
                        default='../expression_prediction/Input/processed_dir/bigwig/',
                        help="Path to the chromatin accessibility data which is a bigwig file ")


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

    coords = parsebed(args.bedpe, lower=args.lower, upper=args.upper, res=args.resolution)#取标记的每对loop的start位置，并在每条染色体上进行排序，生成一个字典包含23条染色体上的所有正样本的两个start位置（除以了10000）
    kde, lower, long_start, long_end = learn_distri_kde(coords)

    # train model per chromosome
    positive_class = {}
    positive_node = {}
    positive_edge = {}
    # positive_seq = {}

    negative_class = {}
    negative_node = {}
    negative_edge = {}
    # negative_seq = {}

    positive_labels = {}
    negative_labels = {}

    exceptional_key = {
        'GM12878': ['18'],
        'K562': ['3', '4', '9', '15', '18'],
        'HCT116': ['13', '17'],
        'IMR90': ['7'],
        'HepG2': ['10'],
        'WTC11': ['6']
    }
    if args.norm_type == 'KR':
        # chromosomes = [str(i) for i in range(1, 23) if i != 18]
        chromosomes = [str(i) for i in range(1, 23) if str(i) not in exceptional_key[args.cell_type]]
    else:
        chromosomes = [str(i) for i in range(1, 23)]
    # chromosomes = [str(i) for i in range(2, 3)]
    for key in chromosomes:
        # if key != '17':
        #     continue
        if key.startswith('chr'):
            chromname = key
        else:
            chromname = 'chr'+key
        # seq_embedding = np.load(args.seq_embed_dir + f'{chromname}.npz')['data']
        # print(seq_embedding.shape)
        print('collecting from {}'.format(key))
        file_path = f'./data/csr_matrix/{args.data_type}_{args.norm_type}/{args.cell_type}/{chromname}.npz'

        print(f"loading matrix from {file_path}")
        X = load_npz(file_path)

        clist = coords[chromname]
        print('okk')

        for (x, epi_node, epi_edge) in generateATAC_woseq(X, clist, chromname, file_dir=os.path.join(args.bigwig_dir, args.cell_type), epi_type=args.epi,
                                                             resou=args.resolution, width=args.width):

            if chromname not in positive_class:
                positive_class[chromname] = []
            if chromname not in positive_node:
                positive_node[chromname] = []

            x = x.astype(np.float32)
            epi_node = epi_node.astype(np.float32)

            positive_class[chromname].append(x)
            positive_node[chromname].append(epi_node)


        positive_num = len((positive_class[chromname]))
        positive_labels[chromname] = np.ones(positive_num).astype(np.int32).tolist()

        neg_coords = negative_generating(X, kde, clist, lower, long_start, long_end)
        stop = len(clist)

        for (x, epi_node, epi_edge) in generateATAC_woseq(X, neg_coords, chromname, file_dir=os.path.join(args.bigwig_dir, args.cell_type), epi_type=args.epi, resou=args.resolution, width=args.width, positive=False,stop=stop):
            if chromname not in negative_class:
                negative_class[chromname] = []
            if chromname not in negative_node:
                negative_node[chromname] = []

            x = x.astype(np.float32)
            epi_node = epi_node.astype(np.float32)

            negative_class[chromname].append(x)
            negative_node[chromname].append(epi_node)


        negative_num = len(negative_class[chromname])
        negative_labels[chromname] = np.zeros(negative_num).astype(np.int32).tolist()

        # np.savez(args.out_dir+'%s_positive.npz' % chromname, data=positive_class[chromname],
        #          node=positive_node[chromname], label=positive_labels[chromname])
        # np.savez(args.out_dir+'%s_negative.npz' % chromname, data=negative_class[chromname],
        #          node=negative_node[chromname], label=negative_labels[chromname])
        np.savez_compressed(args.out_dir+'%s_positive.npz' % chromname, data=positive_class[chromname],
                 node=positive_node[chromname], label=positive_labels[chromname])
        np.savez_compressed(args.out_dir+'%s_negative.npz' % chromname, data=negative_class[chromname],
                 node=negative_node[chromname], label=negative_labels[chromname])

if __name__ == "__main__":
    set_seed(42)
    main()
