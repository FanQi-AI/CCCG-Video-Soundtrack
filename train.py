"""
This code is released and maintained by:

Ke Chen, Yusong Wu, Haohe Liu
MusicLDM: Enhancing Novelty in Text-to-Music Generation Using Beat-Synchronous Mixup Strategies
All rights reserved

contact: knutchen@ucsd.edu
"""

import os
import argparse
import yaml
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from torch.utils.data import DataLoader
from src.latent_diffusion.models.musicldm import MusicLDM
from src.data.dataset import MusicDataset


def main(config, seed):

    seed_everything(seed)

    # 设置缓存
    os.makedirs(config['cache_location'], exist_ok=True)
    os.environ['TRANSFORMERS_CACHE'] = config['cache_location']
    torch.hub.set_dir(config['cache_location'])

    # 设置日志路径
    os.makedirs(config['log_directory'], exist_ok=True)
    log_path = os.path.join(config['log_directory'], os.getlogin())
    os.makedirs(log_path, exist_ok=True)
    folder_name = os.listdir(log_path)
    i = 0
    while str(i) in folder_name:
        i += 1
    log_path = os.path.join(log_path, str(i))
    os.makedirs(log_path, exist_ok=True)

    print(f"Logs and checkpoints will be saved at {log_path}")

    # 加载数据集
    train_dataset = MusicDataset(
        config=config,
        split="train",
    )
    val_dataset = MusicDataset(
        config=config,
        split="val",
    )

    batch_size = config["model"]["params"]["batchsize"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化模型
    latent_diffusion = MusicLDM(**config["model"]["params"])
    latent_diffusion.set_log_dir(log_path, log_path, log_path)

    # Checkpoint 保存策略
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_path, "checkpoints"),
        monitor="val/loss",
        mode="min",
        filename="checkpoint-val_loss={val/loss:.4f}-epoch={epoch}",
        save_top_k=3,
        save_last=True,
    )

    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Lightning Trainer 配置：单机多卡
    trainer = Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=config["trainer"]["max_epochs"],
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=config["trainer"]["val_every_n_epoch"],
        log_every_n_steps=10,
    )

    # 启动训练
    trainer.fit(latent_diffusion, train_loader, val_loader)

    print(f"Training Finished. Check outputs at {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="musicldm.yaml",
        help="Path to the config file"
    )
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    main(config, args.seed)
