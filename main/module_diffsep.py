from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
import wandb
from audio_data_pytorch.utils import fractional_random_split
from audio_diffusion_pytorch import (
    DiffusionSampler,
    Distribution,
    Sampler,
    Schedule,
    UNet1d,
)
from audio_diffusion_pytorch.diffusion import pad_dims
from audio_diffusion_pytorch.utils import default, exists
from einops import rearrange, reduce
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from torch import LongTensor, Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

""" Model """


class Diffusion(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        *,
        sigma_distribution: Distribution,
        sigma_data: float,  # data distribution standard deviation
    ):
        super().__init__()

        self.net = net
        self.sigma_data = sigma_data
        self.sigma_distribution = sigma_distribution

    def get_scale_weights(self, sigmas: Tensor) -> Tuple[Tensor, ...]:
        sigma_data = self.sigma_data
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")
        c_skip = (sigma_data ** 2) / (sigmas_padded ** 2 + sigma_data ** 2)
        c_out = (
            sigmas_padded * sigma_data * (sigma_data ** 2 + sigmas_padded ** 2) ** -0.5
        )
        c_in = (sigmas_padded ** 2 + sigma_data ** 2) ** -0.5
        c_noise = torch.log(sigmas) * 0.25
        return c_skip, c_out, c_in, c_noise

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        batch, device = x_noisy.shape[0], x_noisy.device
        assert exists(sigmas) ^ exists(sigma), "Either sigmas or sigma must be provided"
        # If sigma provided use the same for all batch items (used for sampling)
        if exists(sigma):
            sigmas = torch.full(size=(batch,), fill_value=sigma).to(device)  # type: ignore
        # Predict network output and add skip connection
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas)  # type: ignore
        x_pred = self.net(c_in * x_noisy, c_noise, **kwargs)
        x_denoised = c_skip * x_noisy + c_out * x_pred
        return x_denoised.clamp(-1.0, 1.0)

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return (sigmas ** 2 + self.sigma_data ** 2) * (sigmas * self.sigma_data) ** -2

    def loss(self, x: Tensor, y: Tensor, sigmas: Tensor):
        # Compute weighted loss
        losses = F.mse_loss(x, y, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")
        losses = losses * self.loss_weight(sigmas)
        return losses.mean()

    def forward(
        self, x: Tensor, sigmas: Optional[Tensor] = None, **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x_noisy = x

        if not exists(sigmas):
            # If sigmas not provided, we assume that x has no noise hence we add it
            batch, device = x.shape[0], x.device
            # Sample amount of noise to add for each batch element
            sigmas = self.sigma_distribution(num_samples=batch, device=device)
            sigmas_padded = rearrange(sigmas, "b -> b 1 1")
            noise = torch.randn_like(x)
            x_noisy = x + sigmas_padded * noise

        x_denoised = self.denoise_fn(x_noisy, sigmas=sigmas, **kwargs)

        return x_denoised, sigmas


class Model(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        beta1: float,
        beta2: float,
        in_channels: int,
        channels: int,
        patch_size: int,
        resnet_groups: int,
        kernel_multiplier_downsample: int,
        kernel_sizes_init: Sequence[int],
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
        diffusion_sigma_distribution: Distribution,
        diffusion_sigma_data: int,
        diffusion_dynamic_threshold: float,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

        unet = UNet1d(
            in_channels=in_channels,
            channels=channels,
            patch_size=patch_size,
            resnet_groups=resnet_groups,
            kernel_multiplier_downsample=kernel_multiplier_downsample,
            kernel_sizes_init=kernel_sizes_init,
            multipliers=multipliers,
            factors=factors,
            num_blocks=num_blocks,
            attentions=attentions,
            attention_heads=attention_heads,
            attention_features=attention_features,
            attention_multiplier=attention_multiplier,
            use_attention_bottleneck=use_attention_bottleneck,
            use_nearest_upsample=use_nearest_upsample,
            use_skip_scale=use_skip_scale,
            context_channels=[in_channels + 1],
        )

        self.diffusion = Diffusion(
            net=unet,
            sigma_distribution=diffusion_sigma_distribution,
            sigma_data=diffusion_sigma_data,
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, length, device = x.shape[0], x.shape[2], x.device

        if batch % 2 == 1:
            s0, s1 = x, x
        else:
            # First half of batch is source 0, other half source 1
            s0, s1 = torch.chunk(x, chunks=2, dim=0)
        mix = s0 + s1

        mode_shape = (batch // 2, 1, length)
        mode_separation = torch.zeros(mode_shape, device=device)
        mode_generation = torch.ones(mode_shape, device=device)

        # Guided separation on mixed data
        context = [torch.cat([mix, mode_separation], dim=1)]
        s0_denoised, sigmas = self.diffusion(s0, context=context)
        loss_separation = self.diffusion.loss(s0, s0_denoised, sigmas=sigmas)

        # Unguided serparation on raw data
        with torch.no_grad():
            batch = s0.shape[0]
            context = [torch.cat([s0, mode_separation], dim=1)]
            sigmas = torch.ones(batch, device=device)
            sigmas_padded = rearrange(sigmas, "b -> b 1 1")
            noise = sigmas_padded * torch.randn_like(s0)
            s01_denoised, _ = self.diffusion(noise, sigmas=sigmas, context=context)

        # Condition on s01 to generate s0
        context = [torch.cat([s01_denoised, mode_generation], dim=1)]
        s0_gen_denoised, sigmas = self.diffusion(s0, context=context)
        loss_generation = self.diffusion.loss(s0, s0_gen_denoised, sigmas=sigmas)

        return loss_separation + loss_generation

    def separate(
        self,
        x: Tensor,
        num_steps: int,
        sigma_schedule: Schedule,
        sampler: Sampler,
    ) -> Tensor:
        diffusion_sampler = DiffusionSampler(
            diffusion=self.diffusion,
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            num_steps=num_steps,
        )
        noise = torch.randn_like(x)
        mode_separation = torch.zeros_like(x)
        context = [torch.cat([x, mode_separation], dim=1)]
        return diffusion_sampler(noise, context=context)

    def generate(
        self,
        x: Tensor,
        num_steps: int,
        sigma_schedule: Schedule,
        sampler: Sampler,
    ) -> Tensor:
        diffusion_sampler = DiffusionSampler(
            diffusion=self.diffusion,
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            num_steps=num_steps,
        )
        noise = torch.randn_like(x)
        mode_generation = torch.ones_like(x)
        context = [torch.cat([x, mode_generation], dim=1)]
        return diffusion_sampler(noise, context=context)

    def training_step(self, batch, batch_idx):
        waveforms = batch
        loss = self(waveforms)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms = batch
        loss = self(waveforms)
        self.log("valid_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.parameters()),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        return optimizer

    @property
    def device(self):
        return next(self.diffusion.net.parameters()).device


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
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
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
    logger: WandbLogger, samples: Tensor, sampling_rate: int, comment: str
):
    num_items = samples.shape[0]
    samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()
    logger.log(
        {
            f"sample_{idx}_{comment}": wandb.Audio(
                samples[idx],
                sample_rate=sampling_rate,
            )
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
        sampling_steps: List[int],
        diffusion_schedule: Schedule,
        diffusion_sampler: Sampler,
    ) -> None:
        self.num_items = num_items
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.length = length
        self.sampling_steps = sampling_steps
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
        model = pl_module

        # Log ground truth audio
        x_true = batch[0 : self.num_items]

        log_wandb_audio_batch(
            logger=wandb_logger,
            samples=x_true,
            sampling_rate=self.sampling_rate,
            comment="true",
        )

        for steps in self.sampling_steps:

            # Log generation
            samples = model.generate(
                x=x_true,
                sampler=self.diffusion_sampler,
                sigma_schedule=self.diffusion_schedule,
                num_steps=steps,
            )

            log_wandb_audio_batch(
                logger=wandb_logger,
                samples=samples,
                sampling_rate=self.sampling_rate,
                comment=f"{steps}_gen",
            )

            # Log separation
            samples = model.separate(
                x=x_true,
                sampler=self.diffusion_sampler,
                sigma_schedule=self.diffusion_schedule,
                num_steps=steps,
            )

            log_wandb_audio_batch(
                logger=wandb_logger,
                samples=samples,
                sampling_rate=self.sampling_rate,
                comment=f"{steps}_sep",
            )

        if is_train:
            pl_module.train()
