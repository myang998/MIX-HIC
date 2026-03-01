# Multimodal 3D Genome Pre-training

This repository contains the official implementation of the paper **"Multimodal 3D Genome Pre-training"**, which has been accepted by **NeurIPS 2025**.

📄 **Paper:** [Link](https://openreview.net/pdf?id=jReV7xyjXy)

## Data and Pre-train Weights
Due to the massive volume of the data, we are unable to provide all raw data directly. You can find the sources of the raw data in the **Implementation Details** of our manuscript's **Supplementary Material**. 

We have open-sourced the **raw data processing code** in this repository. You can also use these scripts to process your own datasets. For convenience, we provide the **processed downstream task datasets** and **pre-trained weights** for download:

*   **[Data for CAGE-seq Prediction](https://pan.baidu.com/s/1YVZmrjJFAQo4YMHrPMuXZA?pwd=tghy)**: Download the files and unzip them in `expression_prediction/multimodal/training_data`.
*   **[Data for Loop Detection and Hi-C Contact Map Prediction](https://pan.baidu.com/s/1Se1NrBrJEWV2hJciea_Dqw?pwd=gers)**: Download the files and unzip them in `loop_detection/training_data`.
*   **[Pre-trained Weights](https://pan.baidu.com/s/1NrGEQFb7-SlqJ7rw61kuSA?pwd=xqpz)**: Download the weight files and unzip them in `loop_detection/checkpoints/pretrain`.

## Pipeline and Data Preparation

### i. Hi-C Matrix Normalization
Place your downloaded `.hic` files into the `loop_detection/data/hic/` directory (e.g., `loop_detection/data/hic/K562_4DNFITUOMFUQ.hic`). Then, run `loop_detection/normalize_hic_matrix.py` to perform normalization. The normalized matrices will be saved in the `loop_detection/data/csr_matrix` directory.

### ii. Epigenomic Track Processing
1. Download the raw BAM files (CAGE-seq, ATAC, and DNase) for the corresponding cell lines from the [ENCODE portal](https://www.encodeproject.org/). Use `bash expression_prediction/bam2bw.sh` to convert BAM files into BigWig format. **Note:** This process requires `samtools` and `deepTools` to be installed.
2. Use the `pyBigWig` tool to convert BigWig files into chromosome-specific `.npz` files, and store them in `expression_prediction/Input/processed_dir/bigwig/your_cell_line`.

By following steps **i** and **ii**, you will obtain the pre-processed Hi-C matrices and epigenomic tracks for each chromosome.

---
## Pre-training

### iii. Pre-training and Data Preparation
1. Run `loop_detection/generate_pretrain_data_stage0.py` to create sliding windows across different chromosomes for data segmentation.
2. Run `loop_detection/generate_pretrain_data_stage1.py` to generate paired data consisting of Hi-C matrices and epigenomic tracks.
3. Run `loop_detection/generate_pretrain_data_stage2.py` to merge data from different chromosomes and save them as `memmap` files for efficient loading during pre-training.
4. Run `loop_detection/pretrain.py` to start the pre-training task. This script should be executed using `torchrun` for Distributed Data Parallel (DDP) training.

---

## Downstream Tasks

### iv. Chromatin Loop Detection
1. Download the corresponding loop files (ChIA-PET) and place them in the `loop_detection/data/CHIA-PET` directory. For example, we rename [`ENCFF780PGS.bedpe`](https://www.encodeproject.org/files/ENCFF780PGS/@@download/ENCFF780PGS.bedpe.gz) as `GM12878_CTCF_ENCFF780PGS.txt` for clear definition.
2. Run `loop_detection/generate_downstream_input.py` to generate the paired Hi-C matrix and epigenomic track inputs, along with the corresponding loop ground truth.
3. Run `loop_detection/train_finetune_loop_detection.py` to perform training and fine-tuning for loop detection.
4. Once training is complete, you can use `independent_test_loop_detection.py` for independent testing.

### v. Hi-C Contact Map Prediction
The data preparation and steps for the Hi-C contact map prediction task are identical to step **iv**. Use `train_finetune_map_prediction.py` to perform the training.

### vi. CAGE-Seq Expression Prediction
1. Filter the genes to retain only those within a $\pm$ 250 kb genomic region of the Transcription Start Site (TSS). Save the result as `genelist_filter_GM12878.csv`. Then, use `expression_prediction/multimodal/generate_data_tss_100bp.py` to generate pairwise inputs and the corresponding CAGE-seq expression outputs.
2. Use `expression_prediction/cage_seq_expression_prediction.py` to perform the CAGE-seq expression prediction task.

## Requirements
- Python 3.8.19
- h5py == 3.11.0
- hic-straw == 1.3.1
- einops == 0.8.0
- numpy == 1.24.4
- pillow == 10.4.0
- pyBigWig == 0.3.22
- scikit-learn == 1.3.2
- tokenizers == 0.20.3
- torch == 2.3.0+cu118
- torchaudio == 2.3.0+cu118
- torchdiffeq == 0.2.4
- torchvision == 0.18.0+cu118
- tqdm == 4.65.2
- transformers == 4.46.2