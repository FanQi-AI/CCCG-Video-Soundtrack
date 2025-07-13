'''
This code is released and maintained by:

Ke Chen, Yusong Wu, Haohe Liu
MusicLDM: Enhancing Novelty in Text-to-Music Generation Using Beat-Synchronous Mixup Strategies
All rights reserved

contact: knutchen@ucsd.edu
'''

import os
import sys
import argparse
import yaml
import torch

from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from src.latent_diffusion.models.musicldm import MusicLDM
from src.data.dataset import MusicDataset

def main(config, seed):
    seed_everything(seed)

    os.makedirs(config['cache_location'], exist_ok=True)
    os.environ['TRANSFORMERS_CACHE'] = config['cache_location']
    torch.hub.set_dir(config['cache_location'])

    os.makedirs(config['log_directory'], exist_ok=True)
    log_path = os.path.join(config['log_directory'], os.getlogin())
    os.makedirs(log_path, exist_ok=True)
    folder_name = os.listdir(log_path)
    i = 0
    while str(i) in folder_name:
        i += 1
    log_path = os.path.join(log_path, str(i))
    os.makedirs(log_path, exist_ok=True)

    print(f'Samples will be saved at {log_path}')

    dataset = MusicDataset(
        config=config,
        split="val",
    )
    val_loader = DataLoader(dataset, batch_size=2, num_workers=4, shuffle=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['log_directory'], 'checkpoints'),
        monitor="global_step",
        mode="max",
        filename="checkpoint-fad-{val/frechet_inception_distance:.2f}-global_step={global_step:.0f}",
        every_n_train_steps=1000000,
        save_top_k=1,
        auto_insert_metric_name=False,
        save_last=False,
    )

    latent_diffusion = MusicLDM(**config["model"]["params"])
    latent_diffusion.set_log_dir(log_path, log_path, log_path)

    trainer = Trainer(
        accelerator="gpu",
        devices=[1],
        max_epochs=10,
        num_sanity_val_steps=0,
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=10,
        limit_val_batches=5,
    )

    trainer.test(latent_diffusion, val_loader)

    print(f"Generation Finished. Please check the generation samples and the meta file at {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Generation seed",
        default=0
    )
    args = parser.parse_args()

    config_path = 'musicldm.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    main(config, args.seed)
