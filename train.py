import click
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import get_dataset
from models.model import SemanticNetwork


@click.command()
### Add your options here
@click.option("--config", "-c", type=str, help="path to the config file (.yaml)", default="./config/config.yaml")
@click.option(
    "--weights",
    "-w",
    type=str,
    help="path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.",
    default=None,
)
@click.option(
    "--checkpoint", "-ckpt", type=str, help="path to checkpoint file (.ckpt) to resume training.", default=None
)
def main(config, weights, checkpoint):
    cfg = yaml.safe_load(open(config))
    torch.manual_seed(cfg["experiment"]["seed"])

    # Load data and model
    data = get_dataset(cfg["data"]["name"], cfg)

    if weights is None:
        model = SemanticNetwork(cfg)
    else:
        model = SemanticNetwork.load_from_checkpoint(weights, hparams=cfg, strict=False)

    # Add callbacks:
    checkpoint_saver = ModelCheckpoint(
        dirpath=".",
        monitor="val:miou",
        filename="experiments/" + cfg["experiment"]["id"] + "/best_IoU",
        mode="max",
        verbose=True,
        save_last=True,
    )

    tb_logger = pl_loggers.TensorBoardLogger("experiments/" + cfg["experiment"]["id"], default_hp_metric=False)

    # Setup trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=cfg["train"]["n_gpus"],
        logger=tb_logger,
        num_sanity_val_steps=2,
        accumulate_grad_batches=16,
        resume_from_checkpoint=checkpoint,
        max_epochs=cfg["train"]["max_epoch"],
        callbacks=[checkpoint_saver],
    )
    # Train
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
