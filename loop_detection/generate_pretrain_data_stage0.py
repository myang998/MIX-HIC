import pandas as pd

useful_chr = [f'chr{i}' for i in range(1,23)]
chrom_lens = {}

with open('./data/hg38.chrom.sizes', 'r') as f:
    for line in f.readlines():
        chr, length = line.strip().split('\t')
        if chr in useful_chr:
            chrom_lens[chr] = int(length)


window_size = 250000
step = 62500
with open('data/CHIA-PET/GM12878_created_window.txt', 'w') as f:
    for chr in useful_chr:
        length = chrom_lens[chr]
        for locus in range(window_size//2, length-(window_size//2), step):
            line = f'{chr}\t{locus}\t{locus}\t{chr}\t{locus}\t{locus}\n'
            f.writelines(line)

print('ok')
