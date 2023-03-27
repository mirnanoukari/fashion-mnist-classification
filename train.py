import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from mnist_datamodule import MNISTDataModule
from classifier import Classifier
import hydra

from pytorch_lightning import Trainer
from hydra.utils import instantiate

@hydra.main(config_name="config", config_path=".")
def train(cfg: DictConfig):
    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    optimizer = instantiate(cfg.optimizer, model.parameters(), lr=cfg.lr)
    trainer = Trainer(max_epochs=cfg.epochs, gpus=cfg.gpus)
    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    # Load the configuration file using Hydra
    config = OmegaConf.load("config.yaml")
    train(config)
