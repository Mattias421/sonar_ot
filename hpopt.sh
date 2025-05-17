#!/bin/bash
#SBATCH --partition=dcs-gpu,gpu
#SBATCH --account=dcs-res
#SBATCH --gres=gpu:1
#SBATCH --time=80:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=results/slurm/%x-%a.out
#SBATCH --array=1-16

module load CUDA/12.4.0 

source ./.venv/bin/activate

orion hunt -n $1 -c ./hparams/orion.yaml --version 1 --working-dir ./results/hpopt/ python train.py hparams/train.yaml \
  --hpopt hparams/hpopt.yaml \
  --hpopt_mode orion \
  --scramble_train=true \
  --cfm_sigma~"choices([0.0, 0.0001, 0.001, 0.01, 0.1])" \
  --batch_size~"choices([8, 16, 32, 64, 128])" \
  --weight_decay~"choices([0.0, 1e-5, 1e-4, 1e-3, 1e-2])" \
  --model_layers_per_block~"choices([1, 2, 3, 4, 5, 6])" \
  --model_downsample_each_block~"choices([True, False])" \
  --model_act_fn~"choices(['silu', 'relu', 'swish', 'mish', 'gelu', 'relu'])" --model_norm_num_groups~"choices([4, 8, 16, 32, 64])" 
#--model_block_out_channels~"choices([(64, 128, 256, 256), (32, 64, 128, 128), (16, 32, 64, 64), (128, 256, 512, 512)])" 

