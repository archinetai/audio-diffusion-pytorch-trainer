from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import auraloss
import librosa
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio
import wandb
from audio_data_pytorch.utils import fractional_random_split
from einops import rearrange, reduce
from ema_pytorch import EMA
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from torch import LongTensor, Tensor, nn
from torch.utils.data import DataLoader


def exists(val):
    return val is not None


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_eps: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_weight_decay: float,
        ema_beta: float,
        ema_power: float,
        sample_rate: int,
        autoencoder: nn.Module,
        loss_autoencoder_path: Optional[nn.Module] = None,
        loss_layer: int = 0,
        loss_type: Optional[str] = None,
        loss_bottleneck_weight: Optional[float] = None,
    ):
        super().__init__()
        self.lr = lr
        self.lr_eps = lr_eps
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_weight_decay = lr_weight_decay
        self.sample_rate = sample_rate
        self.loss_type = loss_type
        self.loss_fn = None
        self.loss_bottleneck_weight = loss_bottleneck_weight

        self.autoencoder = autoencoder
        self.autoencoder_ema = EMA(self.autoencoder, beta=ema_beta, power=ema_power)

        if exists(loss_autoencoder_path):
            self.loss_type = "ae"
            self.loss_autoencoder = torch.load(
                loss_autoencoder_path, map_location=self.device
            )
            self.loss_autoencoder.requires_grad_(False)
            self.loss_autoencoder.eval()
            self.loss_layer = loss_layer

    def get_perceptual_features(self, x: Tensor) -> Tensor:
        self.loss_autoencoder.eval()
        _, info = self.loss_autoencoder.encode(x, with_info=True)
        return info["xs"][self.loss_layer]

    def ae_loss(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.get_perceptual_features(x)
        y = self.get_perceptual_features(y)
        return F.mse_loss(x, y)

    def setup(self, stage):
        if self.loss_type == "ae":
            self.loss_fn = self.ae_loss
        elif self.loss_type == "mrstft":
            self.loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
                sample_rate=self.sample_rate,
                device=self.device,
            )
        elif self.loss_type == "sdstft":
            scales = [2048, 1024, 512, 256, 128]
            hop_sizes, win_lengths, overlap = [], [], 0.75
            for scale in scales:
                hop_sizes += [int(scale * (1.0 - overlap))]
                win_lengths += [scale]
            self.loss_fn = auraloss.freq.SumAndDifferenceSTFTLoss(
                fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths
            )
        elif self.loss_type == "mse":
            self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x = batch

        y, info = self.autoencoder(x, with_info=True)
        loss = self.loss_fn(x, y)

        if "loss" in info:
            loss_bottleneck = info["loss"]
            loss += self.loss_bottleneck_weight * loss_bottleneck
            self.log("loss_bottleneck", loss_bottleneck)
        if "perplexity" in info:
            for i, perplexity in enumerate(info["perplexity"]):
                self.log(f"train_perplexity_{i}", perplexity)
        if "replaced_codes" in info:
            for i, replaced_codes in enumerate(info["replaced_codes"]):
                self.log(f"train_replaced_codes_{i}", replaced_codes)

        self.log("train_loss", loss)

        # Update EMA model and log decay
        self.autoencoder_ema.update()
        self.log("ema_decay", self.autoencoder_ema.get_current_decay())
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y = self.autoencoder_ema(x)
        loss = self.loss_fn(x, y)
        self.log("valid_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.parameters()),
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer


""" Datamodule """


class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        *,
        val_split: float,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        **kwargs: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_train: Any = None
        self.data_val: Any = None

    def setup(self, stage: Any = None) -> None:
        split = [1.0 - self.val_split, self.val_split]
        self.data_train, self.data_val = fractional_random_split(self.dataset, split)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )


""" Callbacks """


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    print("WandbLogger not found.")
    return None


def log_wandb_audio_batch(
    logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]
    samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()
    logger.log(
        {
            f"sample_{idx}_{id}": wandb.Audio(
                samples[idx],
                caption=caption,
                sample_rate=sampling_rate,
            )
            for idx in range(num_items)
        }
    )


def log_wandb_audio_spectrogram(
    logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]
    samples = samples.detach().cpu()
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=80,
        center=True,
        norm="slaney",
    )

    def get_spectrogram_image(x):
        spectrogram = transform(x[0])
        image = librosa.power_to_db(spectrogram)
        trace = [go.Heatmap(z=image, colorscale="viridis")]
        layout = go.Layout(
            yaxis=dict(title="Mel Bin (Log Frequency)"),
            xaxis=dict(title="Frame"),
            title_text=caption,
            title_font_size=10,
        )
        fig = go.Figure(data=trace, layout=layout)
        return fig

    logger.log(
        {
            f"mel_spectrogram_{idx}_{id}": get_spectrogram_image(samples[idx])
            for idx in range(num_items)
        }
    )


def log_wandb_embeddings(logger: WandbLogger, id: str, embeddings: Tensor):
    num_items = embeddings.shape[0]
    embeddings = embeddings.detach().cpu()

    def get_figure(x):
        trace = [go.Heatmap(z=x, colorscale="viridis")]
        fig = go.Figure(data=trace)
        return fig

    logger.log(
        {
            f"embedding_{idx}_{id}": get_figure(embeddings[idx])
            for idx in range(num_items)
        }
    )


class SampleLogger(Callback):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        use_ema_model: bool,
    ) -> None:
        self.num_items = num_items
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.length = length
        self.epoch_count = 0
        self.log_next = False
        self.use_ema_model = use_ema_model

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if self.log_next:
            self.log_sample(trainer, pl_module, batch)
            self.log_next = False

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment

        autoencoder = pl_module.autoencoder
        if self.use_ema_model:
            autoencoder = pl_module.autoencoder_ema.ema_model

        x = batch[0 : self.num_items]

        log_wandb_audio_batch(
            logger=wandb_logger,
            id="true",
            samples=x,
            sampling_rate=self.sampling_rate,
        )
        log_wandb_audio_spectrogram(
            logger=wandb_logger,
            id="true",
            samples=x,
            sampling_rate=self.sampling_rate,
        )

        z = autoencoder.encode(x)
        y = autoencoder.decode(z)

        log_wandb_embeddings(logger=wandb_logger, id="z", embeddings=z)
        log_wandb_audio_batch(
            logger=wandb_logger, id="recon", samples=y, sampling_rate=self.sampling_rate
        )
        log_wandb_audio_spectrogram(
            logger=wandb_logger, id="recon", samples=y, sampling_rate=self.sampling_rate
        )

        if is_train:
            pl_module.train()
