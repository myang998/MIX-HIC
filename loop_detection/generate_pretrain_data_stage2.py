#!/usr/bin/env python


import numpy as np
import os
input_dir = './training_data'
output_dir2 = './training_data/pretrain'


chr_list = [f'{i}' for i in range(1, 23)]
# pretrain_cell_type = ['GM12878', 'HCT116', 'IMR90', 'HepG2', 'WTC11']
pretrain_cell_type = ['HCT116', 'IMR90', 'HepG2', 'WTC11']
exceptional_key = {
    'GM12878': ['18'],
    'K562': ['3', '4', '9', '15', '18'],
    'HCT116': ['13', '17'],
    'IMR90': ['7'],
    'HepG2': ['10'],
    'WTC11': ['6']
}


chr_node_memmap = np.memmap(os.path.join(output_dir2, 'all_node2.dat'), dtype=np.float32, mode='w+', shape=(1275948, 5000, 2))
chr_map_memmap = np.memmap(os.path.join(output_dir2, 'all_map2.dat'), dtype=np.float32, mode='w+', shape=(1275948, 50, 50))

idx = 0
for chr in chr_list:
    print('merging chr', chr)
    for cell_type in pretrain_cell_type:
        if chr in exceptional_key[cell_type]:
            continue

        input_file = os.path.join(input_dir, cell_type, 'data_observed_KR_CTCF', f'chr{chr}.npz')
        tmp_file = np.load(input_file)
        node = tmp_file['node']
        matrix = tmp_file['data']
        node_len = len(node)
        chr_node_memmap[idx:idx + node_len] = node
        chr_map_memmap[idx:idx + node_len] = matrix
        idx += len(node)

# output_file = os.path.join(output_dir, 'all.npz')
# np.savez_compressed(output_file, data=chr_data, node=chr_node)

# output_file_matrix = os.path.join(output_dir, 'all_matrix.npz')
# np.savez_compressed(output_file_matrix, data=chr_data)



# all_lenth = 0
# for chr in chr_list:
#     print(chr)
#     total_length = 0
#     for cell_type in pretrain_cell_type:
#         if chr in exceptional_key[cell_type]:
#             continue
#         # input_file = os.path.join(input_dir, cell_type, 'data_observed_KR_map', f'chr{chr}.npz')
#         input_file = os.path.join(input_dir, cell_type, 'data_observed_KR_CTCF', f'chr{chr}.npz')
#         tmp_file = np.load(input_file)
#         node = tmp_file['node']
#         matrix = tmp_file['data']
#
#         node_len = len(node)
#         total_length += node_len
#         # chr_node_memmap[idx:idx + node_len] = node
#         # chr_map_memmap[idx:idx + node_len] = matrix
#         # idx += len(node)
#         print(cell_type, node_len)
#     print(chr, total_length)
#     all_lenth += total_length
# print(all_lenth)
