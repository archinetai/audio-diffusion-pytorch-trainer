from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import librosa
import plotly.graph_objects as go
import plotly.graph_objs as go
import pytorch_lightning as pl
import torch
import torchaudio
import wandb
from audio_data_pytorch.utils import fractional_random_split
from audio_diffusion_pytorch import AudioDiffusionAutoencoder, Sampler, Schedule
from einops import rearrange
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from quantizer_pytorch import Quantizer1d, QuantizerBlock1d, QuantizerChannelwise1d
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
        use_scheduler: bool,
        scheduler_inv_gamma: float,
        scheduler_power: float,
        scheduler_warmup: float,
        encoder_channels: int,
        quantizer_type: str,
        quantizer_loss_weight: float,
        quantizer_codebook_size: int,
        quantizer_expire_threshold: float,
        quantizer_num_residuals: int,
        quantizer_shared_codebook: bool,
        quantizer_ema_decay: float,
        quantizer_groups: int = 1,
        quantizer_split_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.lr = lr
        self.lr_eps = lr_eps
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_weight_decay = lr_weight_decay
        self.use_scheduler = use_scheduler
        self.scheduler_inv_gamma = scheduler_inv_gamma
        self.scheduler_power = scheduler_power
        self.scheduler_warmup = scheduler_warmup
        self.quantizer_loss_weight = quantizer_loss_weight

        if quantizer_type == "timewise":
            self.quantizer = Quantizer1d(
                channels=encoder_channels,
                num_groups=quantizer_groups,
                codebook_size=quantizer_codebook_size,
                expire_threshold=quantizer_expire_threshold,
                num_residuals=quantizer_num_residuals,
                shared_codebook=quantizer_shared_codebook,
                ema_decay=quantizer_ema_decay,
            )
        elif quantizer_type == "channelwise":
            assert_message = "quantizer_split_size required with channelwise type"
            assert quantizer_split_size is not None, assert_message
            self.quantizer = QuantizerChannelwise1d(
                channels=encoder_channels,
                split_size=quantizer_split_size,
                num_groups=quantizer_groups,
                codebook_size=quantizer_codebook_size,
                expire_threshold=quantizer_expire_threshold,
                num_residuals=quantizer_num_residuals,
                shared_codebook=quantizer_shared_codebook,
                ema_decay=quantizer_ema_decay,
            )
        elif quantizer_type == "block":
            assert_message = "quantizer_split_size required with block type"
            assert quantizer_split_size is not None, assert_message
            self.quantizer = QuantizerBlock1d(
                channels=encoder_channels,
                split_size=quantizer_split_size,
                num_groups=quantizer_groups,
                codebook_size=quantizer_codebook_size,
                expire_threshold=quantizer_expire_threshold,
                num_residuals=quantizer_num_residuals,
                shared_codebook=quantizer_shared_codebook,
                ema_decay=quantizer_ema_decay,
            )
        else:
            raise ValueError("Quantizer type must be timewise or channelwise")

        self.model = AudioDiffusionAutoencoder(
            encoder_channels=encoder_channels, bottleneck=self.quantizer, **kwargs
        )

    def from_ids(self, indices: LongTensor) -> Tensor:
        return self.quantizer.from_ids(indices)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x, with_info=True)

    def training_step(self, batch, batch_idx):
        waveforms = batch
        loss, info = self(waveforms)
        self.log("train_loss", loss)
        # Log perplexity of each codebook used
        for i, perplexity in enumerate(info["perplexity"]):
            self.log(f"train_perplexity_{i}", perplexity)
        # Log replaced codes of each codebook used
        for i, replaced_codes in enumerate(info["replaced_codes"]):
            self.log(f"train_replaced_codes_{i}", replaced_codes)
        # Log budget
        for i, budget in enumerate(info["budget"]):
            wandb.log({f"budget_{i}": budget})
        # Log commitment loss
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

        commitment_loss = info["loss"]
        loss += self.quantizer_loss_weight * commitment_loss
        self.log("valid_commitment_loss", commitment_loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.parameters()),
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        if self.use_scheduler:
            scheduler = InverseLR(
                optimizer=optimizer,
                inv_gamma=self.scheduler_inv_gamma,
                power=self.scheduler_power,
                warmup=self.scheduler_warmup,
            )
            return [optimizer], [scheduler]
        return optimizer

    @property
    def device(self):
        return next(self.parameters()).device


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


class QuantizationInfoLogger(Callback):
    def __init__(
        self,
        sample_rate: int,
        patch_blocks: int,
        patch_factor: int,
        split_size: int,
        num_residuals: int,
        num_groups: int,
        quantizer_type: str,
        downsample_factors: List[int],
        encoder_depth: int,
        channels: int,
    ):
        downsample_factors = downsample_factors[0:encoder_depth]
        encoder_downsample = reduce((lambda x, y: x * y), downsample_factors)
        downsample_factor = (patch_factor ** patch_blocks) * encoder_downsample

        self.quantizer_type = quantizer_type
        self.tokens_per_second = 0.0

        self.splits_per_second = 0.0
        if quantizer_type == "channelwise":
            splits_per_second = sample_rate / (downsample_factor * split_size)
            self.splits_per_second = splits_per_second
            self.tokens_per_second = splits_per_second * channels * num_residuals

        self.ticks_per_second = 0.0
        if quantizer_type == "timewise":
            ticks_per_second = sample_rate / downsample_factor
            self.ticks_per_second = ticks_per_second
            self.tokens_per_second = ticks_per_second * num_residuals * num_groups

        if quantizer_type == "block":
            splits_per_second = sample_rate / (downsample_factor * split_size)
            self.tokens_per_second = splits_per_second * num_residuals

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer)
        if self.quantizer_type == "timewise":
            logger.log_hyperparams(
                {
                    "ticks_per_second": self.ticks_per_second,
                    "tokens_per_second": self.tokens_per_second,
                }
            )
        else:
            logger.log_hyperparams(
                {
                    "splits_per_second": self.splits_per_second,
                    "tokens_per_second": self.tokens_per_second,
                }
            )


class SampleLogger(Callback):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        sampling_steps: List[int],
        diffusion_schedule: Schedule,
        diffusion_sampler: Sampler,
    ) -> None:
        self.num_items = num_items
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.length = length
        self.sampling_steps = sampling_steps
        self.epoch_count = 0

        self.diffusion_schedule = diffusion_schedule
        self.diffusion_sampler = diffusion_sampler

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
        model = pl_module.model

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

        context, info = model.encode(x_true, with_info=True)
        indices = info["indices"]
        indices_cpu = rearrange(indices, "b ... -> b (...)").detach().cpu().numpy()

        # Log indices table
        table = go.Figure(data=[go.Table(cells=dict(values=indices_cpu))])
        wandb_logger.log({"indices": table})

        # Compute context from indices (just to make sure it's working)
        context = pl_module.from_ids(indices)

        for steps in self.sampling_steps:

            samples = model.decode(
                context,
                sampler=self.diffusion_sampler,
                sigma_schedule=self.diffusion_schedule,
                num_steps=steps,
            )
            log_wandb_audio_batch(
                logger=wandb_logger,
                id="recon",
                samples=samples,
                sampling_rate=self.sampling_rate,
                caption=f"Sampled in {steps} steps",
            )
            log_wandb_audio_spectrogram(
                logger=wandb_logger,
                id="recon",
                samples=samples,
                sampling_rate=self.sampling_rate,
                caption=f"Sampled in {steps} steps",
            )

        if is_train:
            pl_module.train()
