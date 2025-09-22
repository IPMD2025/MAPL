# MULTI-GRANULARITY ATTRIBUTE PROMPT LEARNING FOR CLOTH-CHANGING PERSON RE-IDENTIFICATION

> Official PyTorch implementation of ["MULTI-GRANULARITY ATTRIBUTE PROMPT LEARNING FOR CLOTH-CHANGING PERSON RE-IDENTIFICATION"]


### Training

```sh
# LTCC
CUDA_VISIBLE_DEVICES=0 python main.py --gpu_devices 0 --dataset_root /media/data2/lx/cloth-changing/dataset --max_epoch 40 --save_dir ./savefile --save_checkpoint --reranking --ablation featandclo

# PRCC
CUDA_VISIBLE_DEVICES=2 python main.py --gpu_devices 2 --dataset prcc --dataset_root /media/data2/lx/cloth-changing/dataset --dataset_filename PRCC --max_epoch 40 --save_dir ./savefile --save_checkpoint --reranking --ablation featandclo

`--dataset_root` : replace `DATASET_ROOT` with your dataset root path

`--save_dir`: replace `SAVE_DIR` with the path to save log file and checkpoints

The 'PAR-PETA_1220_0.5. txt' in 'PRCC. py' and 'LTCC. py' can be generated through AAPAR（“https://github.com/IPMD2025/AAPAR/blob/main/demo_PETA_ltcc.py”、“https://github.com/IPMD2025/AAPAR/blob/main/demo_PETA_prcc.py”）
