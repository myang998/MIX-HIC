#!/bin/bash

# Prepare epigenomic tracks (.bw format)
# For each epigenomic track: 
# --> download bam file 
# --> merge bam replicates 
# --> sort and index bam file 
# --> normalize and create .bw file
# 
# Usage example: 
# screen
# bash /epiphany/data_preparation/bam2bw.sh \
# https://www.encodeproject.org/files/ENCFF353YPB/@@download/ENCFF353YPB.bam \
# https://www.encodeproject.org/files/ENCFF677MAG/@@download/ENCFF677MAG.bam \
# GM12878_H3K36me3

source /home/yangminghao/anaconda3/bin/activate mixhic


BAM1=$1
BAM2=$2
FILE_NAME1="${2:-"GM12878_1_ENCFF593WBR.bam"}"
FILE_NAME2="${3:-"GM12878_2_ENCFF658WKQ.bam"}"

DOWNLOAD_FOLDER="${5:-"./bam_files/K562"}"
FILE_STORE_FOLDER="${6:-"./generated_files/K562"}"
GENOME_ASSEMBLY="${7:-"hg38"}"
SAVE_FILE_NAME="${9:-"GM12878"}"

export PATH=/packages/samtools-1.17:$PATH

echo "${FILE_NAME1}"
echo "${FILE_NAME2}"


#统计读数

# samtools merge -o "${FILE_STORE_FOLDER}"/"${SAVE_FILE_NAME}"_merged.bam "${DOWNLOAD_FOLDER}"/"${FILE_NAME1}"
samtools merge -o "${FILE_STORE_FOLDER}"/"${SAVE_FILE_NAME}"_merged.bam "${DOWNLOAD_FOLDER}"/"${FILE_NAME1}" "${DOWNLOAD_FOLDER}"/"${FILE_NAME2}"
# samtools merge -o "${FILE_STORE_FOLDER}"/"${SAVE_FILE_NAME}"_merged.bam "${DOWNLOAD_FOLDER}"/"${FILE_NAME1}" "${DOWNLOAD_FOLDER}"/"${FILE_NAME2}" "${DOWNLOAD_FOLDER}"/"${FILE_NAME3}"

samtools sort -o "${FILE_STORE_FOLDER}"/"${SAVE_FILE_NAME}"_merged_sorted.bam "${FILE_STORE_FOLDER}"/"${SAVE_FILE_NAME}"_merged.bam

samtools index "${FILE_STORE_FOLDER}"/"${SAVE_FILE_NAME}"_merged_sorted.bam


bamCoverage --bam "${FILE_STORE_FOLDER}"/"${SAVE_FILE_NAME}"_merged_sorted.bam \
-o "${FILE_STORE_FOLDER}"/"${SAVE_FILE_NAME}"_merged_sorted_RPGC.bigWig \
--normalizeUsing RPGC \
--binSize 10 \
--effectiveGenomeSize 2913022398 \
--ignoreForNormalization chrX chrM