# Trainer for [`audio-diffusion-pytorch`](https://github.com/archinetai/audio-diffusion-pytorch)

audio-diffusion-pytorch-trainer notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/archinetai/audio-diffusion-pytorch-trainer/blob/main/notebooks/audio_diffusion_pytorch_trainer_v0_2.ipynb)

## Setup

(Optional) Create virtual environment and activate it

```bash
python3 -m venv venv

source venv/bin/activate
```
Install requirements

```bash
pip install -r requirements.txt
```

Add environment variables, rename `.env.tmp` to `.env` and replace with your own variables (example values are random)
```bash
DIR_LOGS=/logs
DIR_DATA=/data

# Required if using wandb logger
WANDB_PROJECT=audioproject
WANDB_ENTITY=johndoe
WANDB_API_KEY=a21dzbqlybbzccqla4txa21dzbqlybbzccqla4tx

# Required if using Common Voice dataset
HUGGINGFACE_TOKEN=hf_NUNySPyUNsmRIb9sUC4FKR2hIeacJOr4Rm
```

## Run Experiments
Run test experiment, see the [`exp`](exp/) folder for other experiments (create your own `.yaml` file there to run a custom experiment!)
```bash
python train.py exp=base_test
```

Run on GPU(s)

```bash
python train.py exp=base_test trainer.gpus=1
```

Resume run from a checkpoint

```bash
python train.py exp=base_test +ckpt=/logs/ckpts/2022-08-17-01-22-18/'last.ckpt'
```

## FAQ

<details>
<summary>How do I use the CommonVoice dataset?</summary>

Before running an experiment on commonvoice dataset you have to:
1. Create a Huggingface account if you don't already have one [here](https://huggingface.co/join)
2. Accept the terms of the version of [common voice dataset](https://huggingface.co/mozilla-foundation) you will be using by clicking on it and selecting "Access repository".
3. Add your [access token](https://huggingface.co/settings/tokens) to the `.env` file, for example `HUGGINGFACE_TOKEN=hf_NUNySPyUNsmRIb9sUC4FKR2hIeacJOr4Rm`.

</details>

<details>
<summary>How do I load the model once I'm done training?</summary>

If you want to load the checkpoint to restore training with the trainer you can do `python train.py exp=my_experiment +ckpt=/logs/ckpts/2022-08-17-01-22-18/'last.ckpt'`.

Otherwise if you want to instantiate a model from the checkpoint:
```py
from main.mymodule import Model
model = Model.load_from_checkpoint(
    checkpoint_path='my_checkpoint.ckpt',
    learning_rate=1e-4,
    beta1=0.9,
    beta2=0.99,
    in_channels=1,
    patch_size=16,
    all_other_paratemeters_here...
)
```
to get only the PyTorch `.pt` checkpoint you can save the internal model weights as `torch.save(model.model.state_dict(), 'torchckpt.pt')`.

</details>


<details>
<summary>Why no checkpoint is created at the end of the epoch?</summary>

If the epoch is shorter than `log_every_n_steps` it doesn't save the checkpoint at the end of the epoch, but after the provided number of steps. If you want to checkpoint more frequently you can add `every_n_train_steps` to the ModelCheckpoint e.g.:
```yaml
model_checkpoint:
    _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: "valid_loss"   # name of the logged metric which determines when model is improving
    save_top_k: 1           # save k best models (determined by above metric)
    save_last: True         # additionaly always save model from last epoch
    mode: "min"             # can be "max" or "min"
    verbose: False
    dirpath: ${logs_dir}/ckpts/${now:%Y-%m-%d-%H-%M-%S}
    filename: '{epoch:02d}-{valid_loss:.3f}'
    every_n_train_steps: 10
```
Note that logging the checkpoint so frequently is not recommended in general, since it takes a bit of time to store the file.

</details>
