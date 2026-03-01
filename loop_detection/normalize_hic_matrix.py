import argparse
import numpy as np
from dataUtils_all_physico import *
from utils import *
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

    parser.add_argument("-p", dest="path", type=str, default='./data/hic/K562_4DNFITUOMFUQ.hic',
                        help="Path to a .cool URI string or a .hic file.")
    parser.add_argument("-c", dest="cell_type", default='K562', type=str, help="cell line")

    # parser.add_argument("-n", dest="norm_type", type=str, default='KR')
    # parser.add_argument("-t", dest="data_type", type=str, default='observed')

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
    # import hicstraw
    # hicstraw.straw('oe', 'NONE', './data/hic/GM12878_4DNFI1UEG1HD.hic', '22', '22',
    #             'BP', 5000)
    # print('ok')

    args = get_args()
    np.seterr(divide='ignore', invalid='ignore')

    chromosomes = [str(i) for i in range(1, 23)]
    print('creating matrix from ', chromosomes)
    for key in chromosomes:
        if key.startswith('chr'):
            chromname = key
        else:
            chromname = 'chr'+key

        print('collecting from {}'.format(key))
        file_path = f'./data/csr_matrix/{args.data_type}_{args.norm_type}/{args.cell_type}/{chromname}.npz'

        if args.norm_type == 'KR':
            X = csr_contact_matrix(args.norm_type, args.path, key, key, 'BP', args.resolution, data_type=args.data_type, rescale=False)
        elif args.norm_type == 'NONE':
            X = csr_contact_matrix(args.norm_type, args.path, key, key, 'BP', args.resolution, data_type=args.data_type,
                                   clamp=False, rescale=False)
        save_npz(file_path, X)
        print(f"saving matrix to {file_path}")


if __name__ == "__main__":
    set_seed(42)
    main()


# Data1 = np.load(f'./data/csr_matrix/observed_KR/GM12878_tmp/chr22.npz')['data']
# Data2 = np.load(f'./data/csr_matrix/observed_KR/GM12878/chr22.npz')['data']
# print('kk')