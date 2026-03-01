import pickle,os,h5py
import numpy as np
from scipy.sparse import load_npz,csr_matrix,save_npz
import torch
def pad_seq_matrix(matrix, pad_len=300):
    # add flanking region to each sample
    paddings = np.zeros((1, 4, pad_len)).astype('int8')
    dmatrix = np.concatenate((paddings, matrix[:, :, -pad_len:]), axis=0)[:-1, :, :]
    umatrix = np.concatenate((matrix[:, :, :pad_len], paddings), axis=0)[1:, :, :]
    return np.concatenate((dmatrix, matrix, umatrix), axis=2)

def pad_signal_matrix(matrix, pad_len=300):
    paddings = np.zeros(pad_len).astype('float32')
    dmatrix = np.vstack((paddings, matrix[:, -pad_len:]))[:-1, :]
    umatrix = np.vstack((matrix[:, :pad_len], paddings))[1:, :]
    return np.hstack((dmatrix, matrix, umatrix))

def load_ref_genome(chr):
    ref_path = '../Input/processed_dir/one_hot/'
    print(chr)
    ref_file = os.path.join(ref_path, 'chr%s.npz' % chr)
    ref_gen_data = load_npz(ref_file).toarray()
    ref_gen_data = ref_gen_data.reshape(4, -1, 1000).swapaxes(0, 1)
    return torch.tensor(pad_seq_matrix(ref_gen_data))

def load_dnase(dnase_seq):
    dnase_seq = np.expand_dims(pad_signal_matrix(dnase_seq.reshape(-1, 1000)), axis=1)
    return torch.tensor(dnase_seq)

def load_cage(cl):
    # cage_file='cage/data/%s_seq_cov.h5'%cl
    cage_file = 'cage/data/%s_seq_cov_50win.h5' % cl
    with h5py.File(cage_file, 'r') as hf:
        cage_data = np.array(hf['targets'])
    return cage_data

def prepare_train_data(cls):
    """
        dnase: (1, len(chrom))
        ref sequence one hot data: (len(chrom) / binsize, 4, binsize + 600), where binsize = 1000
        cage_data: (13405, 250) , num_seq, seq_len. In sequences.bed file, the length is 250k bp, binsize=1000
    """
    dnase_data={}
    ref_data={}
    cage_data={}
    chroms=[str(i) for i in range(1,23)]
    for chr in chroms:
        ref_data[chr] = load_ref_genome(chr)
    for cl in cls:
        cage_data[cl]=load_cage(cl)
        dnase_data[cl]={}
        dnase_path = '../Input/processed_dir/'
        with open(dnase_path + '%s_DNase_merged_sorted_RPGC.pickle' % cl, 'rb') as f:
            dnase = pickle.load(f)
        for chr in range(1,23):
            dnase_data[cl][str(chr)]=load_dnase(csr_matrix(dnase[chr]).toarray())
    return dnase_data, ref_data,cage_data

def prepare_train_data_multimodal(cls, data_dir, except_chroms=None):
    """
        dnase: (1, len(chrom))
        ref sequence one hot data: (len(chrom) / binsize, 4, binsize + 600), where binsize = 1000
        cage_data: (13405, 250) , num_seq, seq_len. In sequences.bed file, the length is 250k bp, binsize=1000
    """
    # matrix_data={}
    # epi_data={}
    # cage_data={}
    matrix_data = []
    epi_data = []
    cage_data = []

    chroms = [f'chr{i}' for i in range(1,23)]
    if except_chroms != None:
        chroms = [chrom for chrom in chroms if chrom not in except_chroms]
    for chr in chroms:
        data = np.load(os.path.join(data_dir, f'{chr}_positive.npz'))
        matrix = data['data']
        epi = data['node']
        label = data['label']

        # matrix_data[chr] = matrix
        # epi_data[chr] = epi
        # cage_data[chr] = label
        matrix_data.extend(matrix)
        epi_data.extend(epi)
        cage_data.extend(label)
    return np.array(matrix_data), np.array(epi_data), np.array(cage_data)

def prepare_pretrain_data(data_dir, chroms):
    """
        dnase: (1, len(chrom))
        ref sequence one hot data: (len(chrom) / binsize, 4, binsize + 600), where binsize = 1000
        cage_data: (13405, 250) , num_seq, seq_len. In sequences.bed file, the length is 250k bp, binsize=1000
    """

    matrix_data = []
    epi_data = []
    cage_data = []

    for chr in chroms:
        data = np.load(os.path.join(data_dir, f'{chr}_positive.npz'))
        matrix = data['data']
        epi = data['node']
        label = data['label']

        matrix_data.extend(matrix)
        epi_data.extend(epi)
        cage_data.extend(label)
    return np.array(matrix_data), np.array(epi_data), np.array(cage_data)


def prepare_finetune_data(data_dir, chroms, except_chroms=None):
    """
        dnase: (1, len(chrom))
        ref sequence one hot data: (len(chrom) / binsize, 4, binsize + 600), where binsize = 1000
        cage_data: (13405, 250) , num_seq, seq_len. In sequences.bed file, the length is 250k bp, binsize=1000
    """
    # matrix_data={}
    # epi_data={}
    # cage_data={}
    matrix_data = []
    epi_data = []
    cage_data = []

    if except_chroms != None:
        chroms = [chrom for chrom in chroms if chrom not in except_chroms]
    for chr in chroms:
        data = np.load(os.path.join(data_dir, f'{chr}_positive.npz'))
        matrix = data['data']
        epi = data['node']
        label = data['label']

        # matrix_data[chr] = matrix
        # epi_data[chr] = epi
        # cage_data[chr] = label
        matrix_data.extend(matrix)
        epi_data.extend(epi)
        cage_data.extend(label)
    return np.array(matrix_data), np.array(epi_data), np.array(cage_data)