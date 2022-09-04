import os

import dotenv
import hydra
import pytorch_lightning as pl
from main import utils
from omegaconf import DictConfig

# Load environment variables from `.env`.
dotenv.load_dotenv(override=True)
log = utils.get_logger(__name__)


@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(config: DictConfig) -> None:

    # Logs config tree
    utils.extras(config)

    # Apply seed for reproducibility
    pl.seed_everything(config.seed)

    # Initialize datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>.")
    datamodule = hydra.utils.instantiate(config.datamodule, _convert_="partial")

    # Initialize model
    log.info(f"Instantiating model <{config.model._target_}>.")
    model = hydra.utils.instantiate(config.model)

    # Initialize all callbacks (e.g. checkpoints, early stopping)
    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>.")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Initialize loggers (e.g. wandb)
    loggers = []
    if "loggers" in config:
        for _, lg_conf in config["loggers"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>.")
                # Sometimes wandb throws error if slow connection...
                logger = utils.retry_if_error(lambda: hydra.utils.instantiate(lg_conf))
                loggers.append(logger)

    # Initialize trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>.")
    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Train with checkpoint if present, otherwise from start
    if "ckpt" in config:
        ckpt = config.get("ckpt")
        log.info(f"Starting training from {ckpt}")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)
    else:
        log.info("Starting training.")
        trainer.fit(model=model, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
