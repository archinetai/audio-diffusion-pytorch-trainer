# Trainer for [`audio-diffusion-pytorch`](https://github.com/archinetai/audio-diffusion-pytorch)

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
python train.py exp=youtube_test
```

Run on GPU(s)

```bash
python train.py exp=base_youtube_test trainer.gpus=1
```

Resume run from a checkpoint

```bash
python train.py exp=base_youtube_test +ckpt=/logs/ckpts/2022-08-17-01-22-18/'last.ckpt'
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
