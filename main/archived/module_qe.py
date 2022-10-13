from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import auraloss
import librosa
import plotly.graph_objects as go
import plotly.graph_objs as go
import pytorch_lightning as pl
import torch
import torchaudio
import wandb
from audio_data_pytorch.utils import fractional_random_split
from einops import rearrange
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from torch import LongTensor, Tensor, nn
from torch.utils.data import DataLoader

""" Model """


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_eps: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_weight_decay: float,
        sample_rate: float,
        quantizer_loss_weight: float,
        loss_type: str,
        autoencoder: nn.Module,
    ):
        super().__init__()
        self.lr = lr
        self.lr_eps = lr_eps
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_weight_decay = lr_weight_decay
        self.sample_rate = sample_rate
        self.quantizer_loss_weight = quantizer_loss_weight
        self.loss_type = loss_type

        self.autoencoder = autoencoder
        self.loss_fn = None

    def setup(self, stage):
        if self.loss_type == "mrstft":
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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z, info = self.autoencoder.encode(x, with_info=True)  # type: ignore
        y = self.autoencoder.decode(z)  # type: ignore
        loss = self.loss_fn(x, y)  # type: ignore
        return loss, info

    def training_step(self, batch, batch_idx):
        waveforms = batch
        loss, info = self(waveforms)
        self.log("train_loss", loss)
        # Log perplexity of each codebook used
        for i, perplexity in enumerate(info["perplexity"]):
            self.log(f"train_perplexity_{i}", perplexity)
        # Log replaced codes of each codebook used
        if "replaced_codes" in info:
            for i, replaced_codes in enumerate(info["replaced_codes"]):
                self.log(f"train_replaced_codes_{i}", replaced_codes)
        # Log commitment loss
        if "loss" in info:
            commitment_loss = info["loss"]
            loss += self.quantizer_loss_weight * commitment_loss
            self.log("train_commitment_loss", commitment_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms = batch
        loss, info = self(waveforms)
        self.log("valid_loss", loss)

        for i, perplexity in enumerate(info["perplexity"]):
            self.log(f"valid_perplexity_{i}", perplexity)

        if "loss" in info:
            commitment_loss = info["loss"]
            loss += self.quantizer_loss_weight * commitment_loss
            self.log("valid_commitment_loss", commitment_loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.autoencoder.parameters()),
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            # weight_decay=self.lr_weight_decay,
        )
        return optimizer

    @property
    def device(self):
        return next(self.autoencoder.parameters()).device


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


class SampleLogger(Callback):
    def __init__(
        self, num_items: int, channels: int, sampling_rate: int, length: int
    ) -> None:
        self.num_items = num_items
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.length = length
        self.epoch_count = 0
        self.log_next = False

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

        # Encode x_true to get indices
        x_true = batch[0 : self.num_items]

        log_wandb_audio_batch(
            logger=wandb_logger,
            id="true",
            samples=x_true,
            sampling_rate=self.sampling_rate,
        )
        log_wandb_audio_spectrogram(
            logger=wandb_logger,
            id="true",
            samples=x_true,
            sampling_rate=self.sampling_rate,
        )

        _, info = autoencoder.encode(x_true, with_info=True)
        indices = info["indices"]
        indices_cpu = rearrange(indices, "b ... -> b (...)").detach().cpu().numpy()

        # Log indices table
        table = go.Figure(data=[go.Table(cells=dict(values=indices_cpu))])
        wandb_logger.log({"indices": table})

        # Compute from indices (just to make sure it's working)
        z = autoencoder.bottleneck.from_ids(indices)
        y = autoencoder.decode(z)

        log_wandb_audio_batch(
            logger=wandb_logger, id="recon", samples=y, sampling_rate=self.sampling_rate
        )
        log_wandb_audio_spectrogram(
            logger=wandb_logger, id="recon", samples=y, sampling_rate=self.sampling_rate
        )

        if is_train:
            pl_module.train()
