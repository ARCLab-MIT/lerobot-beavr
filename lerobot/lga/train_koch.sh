python lerobot/scripts/train.py \
  --dataset.repo_id=arclabmit/koch_masked_cubebin_dataset \
  --policy.type=act \
  --output_dir=outputs/train/koch_masked_cubebin \
  --job_name=koch_masked_cubebin \
  --policy.device=cuda \
  --wandb.enable=true