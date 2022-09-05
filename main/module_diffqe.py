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
from audio_diffusion_pytorch import Distribution, Encoder1d, Model1d, Sampler, Schedule
from audio_diffusion_pytorch.modules import ResnetBlock1d, TransformerBlock1d
from einops import rearrange
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from quantizer_pytorch import Quantizer1d
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
        in_channels: int,
        channels: int,
        patch_size: int,
        resnet_groups: int,
        kernel_multiplier_downsample: int,
        kernel_sizes_init: Sequence[int],
        out_means: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        attentions: Sequence[bool],
        attention_heads: int,
        attention_features: int,
        attention_multiplier: int,
        use_nearest_upsample: bool,
        use_skip_scale: bool,
        use_attention_bottleneck: bool,
        extract_channels: List[int],
        context_channels: List[int],
        codebook_size: int,
        quantizer_type: str,
        quantizer_loss_weight: float,
        quantizer_groups: int,
        quantizer_split_size: int,
        quantizer_mask_proba_min: float,
        quantizer_mask_proba_max: float,
        quantizer_mask_proba_rho: float,
        diffusion_sigma_distribution: Distribution,
        diffusion_sigma_data: int,
        diffusion_dynamic_threshold: float,
        quantizer_num_residuals: int = 1,
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
        self.quantizer_type = quantizer_type
        self.quantizer_loss_weight = quantizer_loss_weight

        self.encoder = Encoder1d(
            in_channels=in_channels,
            channels=channels,
            patch_size=patch_size,
            resnet_groups=resnet_groups,
            kernel_multiplier_downsample=kernel_multiplier_downsample,
            kernel_sizes_init=kernel_sizes_init,
            multipliers=multipliers,
            factors=factors,
            num_blocks=num_blocks,
            extract_channels=extract_channels,
        )

        quantizer_channels = extract_channels[-1]
        post_quantizer_channels = context_channels[-1]
        extra_args = (
            {"num_residuals": quantizer_num_residuals}
            if quantizer_type == "rvqe"
            else {}
        )

        self.quantizer = Quantizer1d(
            channels=quantizer_channels,
            split_size=quantizer_split_size,
            num_groups=quantizer_groups,
            quantizer_type=quantizer_type,
            codebook_size=codebook_size,
            mask_proba_min=quantizer_mask_proba_min,
            mask_proba_max=quantizer_mask_proba_max,
            mask_proba_rho=quantizer_mask_proba_rho,
            **extra_args,
        )

        self.post_quantizer = nn.Sequential(
            ResnetBlock1d(
                in_channels=quantizer_channels,
                out_channels=post_quantizer_channels,
                num_groups=resnet_groups,
            ),
            TransformerBlock1d(
                channels=post_quantizer_channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
            ),
            ResnetBlock1d(
                in_channels=post_quantizer_channels,
                out_channels=post_quantizer_channels,
                num_groups=resnet_groups,
            ),
        )

        self.model = Model1d(
            in_channels=in_channels,
            channels=channels,
            patch_size=patch_size,
            resnet_groups=resnet_groups,
            kernel_multiplier_downsample=kernel_multiplier_downsample,
            kernel_sizes_init=kernel_sizes_init,
            out_means=out_means,
            multipliers=multipliers,
            factors=factors,
            num_blocks=num_blocks,
            attentions=attentions,
            attention_heads=attention_heads,
            attention_features=attention_features,
            attention_multiplier=attention_multiplier,
            use_nearest_upsample=use_nearest_upsample,
            use_skip_scale=use_skip_scale,
            use_attention_bottleneck=use_attention_bottleneck,
            diffusion_sigma_distribution=diffusion_sigma_distribution,
            diffusion_sigma_data=diffusion_sigma_data,
            diffusion_dynamic_threshold=diffusion_dynamic_threshold,
            context_channels=[0, 0]
            + context_channels,  # Skip in and post patch channels
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Dict]:
        x = self.encoder(x)[-1]
        x, info = self.quantizer(x)
        x = self.post_quantizer(x)
        return x, info

    def from_ids(self, indices: LongTensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.quantizer.from_ids(indices, mask)
        x = self.post_quantizer(x)
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict]:
        context, info = self.encode(x)
        return self.model(x, context=[context]), info

    def training_step(self, batch, batch_idx):
        waveforms = batch
        loss, info = self(waveforms)
        self.log("train_loss", loss)
        for i, perplexity in enumerate(info["perplexity"]):
            self.log(f"train_perplexity_{i}", perplexity)
        if self.quantizer_type in ["vq", "vqe", "rvqe"]:
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
        if self.quantizer_type in ["vq", "vqe", "rvqe"]:
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
        return next(self.model.parameters()).device


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
        patch_size: int,
        split_size: int,
        num_residuals: int,
        downsample_factors: List[int],
        extract_channels: List[int],
    ):
        encoder_depth = len(extract_channels)
        downsample_factors = downsample_factors[0:encoder_depth]
        encoder_downsample = reduce((lambda x, y: x * y), downsample_factors)
        downsample_factor = patch_size * encoder_downsample * split_size

        splits_per_second = sample_rate / downsample_factor
        self.splits_per_second = splits_per_second

        tokens_per_second = splits_per_second * extract_channels[-1] * num_residuals
        self.tokens_per_second = tokens_per_second

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer)
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

        context, info = pl_module.encode(x_true)
        indices = info["indices"]

        mask = info["mask"]
        indices_cpu = rearrange(indices, "b c s -> b (c s)").detach().cpu().numpy()

        # Log indices table
        table = go.Figure(data=[go.Table(cells=dict(values=indices_cpu))])
        wandb_logger.log({"indices": table})

        # Get start diffusion noise
        batch = x_true.shape[0]
        noise = torch.randn(
            (batch, self.channels, self.length), device=pl_module.device
        )

        # for channels in self.num_channels:

        # # Create mask
        # mask = torch.zeros_like(mask)
        # num_splits = mask.shape[-1]
        # mask[:, :channels, :] = torch.ones(
        #     (batch, channels, num_splits), dtype=torch.bool, device=pl_module.device
        # )

        # Compute context
        context = pl_module.from_ids(indices)

        for steps in self.sampling_steps:

            samples = model.sample(
                noise=noise,
                sampler=self.diffusion_sampler,
                sigma_schedule=self.diffusion_schedule,
                num_steps=steps,
                context=[context],
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
