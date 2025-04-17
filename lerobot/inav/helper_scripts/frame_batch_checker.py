from lerobot.common.datasets import factory
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.datasets.transforms import ImageTransformsConfig
from lerobot.common.datasets.transforms import ImageTransformConfig
from lerobot.common.policies.act.configuration_act import NormalizationMode
from lerobot.configs.default import WandBConfig
from lerobot.configs.default import EvalConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from pathlib import PosixPath
from huggingface_hub import hf_hub_download
import json, pprint

# Create configuration object
cfg = TrainPipelineConfig(
    dataset=DatasetConfig(
        repo_id='aposadasn/lander30',
        root=None,
        episodes=None,
        image_transforms=ImageTransformsConfig(
            enable=False,
            max_num_transforms=3,
            random_order=False,
            tfs={
                'brightness': ImageTransformConfig(weight=1.0, type='ColorJitter', kwargs={'brightness': (0.8, 1.2)}),
                'contrast': ImageTransformConfig(weight=1.0, type='ColorJitter', kwargs={'contrast': (0.8, 1.2)}),
                'saturation': ImageTransformConfig(weight=1.0, type='ColorJitter', kwargs={'saturation': (0.5, 1.5)}),
                'hue': ImageTransformConfig(weight=1.0, type='ColorJitter', kwargs={'hue': (-0.05, 0.05)}),
                'sharpness': ImageTransformConfig(weight=1.0, type='SharpnessJitter', kwargs={'sharpness': (0.5, 1.5)})
            }
        ),
        revision=None,
        use_imagenet_stats=True,
        video_backend='torchcodec'
    ),
    env=None,
    policy=ACTConfig(
        n_obs_steps=1,
        normalization_mapping={
            'VISUAL': NormalizationMode.MEAN_STD,
            'STATE': NormalizationMode.MEAN_STD,
            'ACTION': NormalizationMode.MEAN_STD
        },
        input_features={},
        output_features={},
        device='cuda',
        use_amp=False,
        chunk_size=100,
        n_action_steps=100,
        vision_backbone='resnet18',
        pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1',
        replace_final_stride_with_dilation=False,
        pre_norm=False,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        feedforward_activation='relu',
        n_encoder_layers=4,
        n_decoder_layers=1,
        use_vae=True,
        latent_dim=32,
        n_vae_encoder_layers=4,
        temporal_ensemble_coeff=None,
        dropout=0.1,
        kl_weight=10.0,
        optimizer_lr=1e-05,
        optimizer_weight_decay=0.0001,
        optimizer_lr_backbone=1e-05
    ),
    output_dir=PosixPath('outputs/train/lander'),
    job_name='lander',
    resume=False,
    seed=1000,
    num_workers=4,
    batch_size=8,
    steps=100000,
    eval_freq=20000,
    log_freq=200,
    save_checkpoint=True,
    save_freq=20000,
    use_policy_training_preset=True,
    optimizer=None,
    scheduler=None,
    eval=EvalConfig(
        n_episodes=50,
        batch_size=50,
        use_async_envs=False
    ),
    wandb=WandBConfig(
        enable=True,
        disable_artifact=False,
        project='lerobot',
        entity=None,
        notes=None,
        run_id=None,
        mode=None
    )
)

# Now create the dataset
ds = factory.make_dataset(cfg)
print(ds.meta.video_keys)               # should list "observation.image.cam"
sample = ds[0]
print(sample.keys())

info_path = hf_hub_download('aposadasn/lander30',
                            'meta/info.json',
                            repo_type='dataset')
with open(info_path) as f:
    info = json.load(f)

pprint.pp(info['features'].get('observation.image.cam'))
# → probably None  (or dtype == "image")