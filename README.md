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
python train.py exp=youtube_test trainer.gpus=1
```

Resume run from a checkpoint

```bash
python train.py exp=youtube_test +ckpt=/logs/ckpts/2022-08-17-01-22-18/'last.ckpt'
```
