source ./.venv/bin/activate

orion hunt -n $1 -c ./hparams/orion.yaml --version 1 --working-dir ./results/hpopt/ python train.py hparams/train.yaml \
  --hpopt hparams/hpopt.yaml \
  --hpopt_mode orion \
  --cfm_sigma~"choices([0.0, 0.0001, 0.001, 0.01, 0.1])" \
  --cfm_ode_solver~"choices(['dopri5', 'euler'])" \
  --batch_size~"choices([8, 16, 32, 64, 128])" \
  --learning_rate~"loguniform(1e-5, 1e-2)" \
  --weight_decay~"choices([0.0, 1e-5, 1e-4, 1e-3, 1e-2])" \
  --model_layers_per_block~"choices([1, 2, 3])" \
  --model_downsample_each_block~"choices([True, False])" \
  --model_norm_num_groups~"choices([16, 32, 64])" \ 

