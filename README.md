# GVPO

Official implementation of **Group Verification-based Policy Optimization for Interactive Coding Agents**.

Accepted at **ICLR 2026**.

[Paper on OpenReview](https://openreview.net/forum?id=RY47Tq0VsV&noteId=FUhjFIlSWQ)

## Overview

This repository is the official implementation of GVPO, proposed in *Group Verification-based Policy Optimization for Interactive Coding Agents*.

This repository is built on top of the `verl` framework and extends it with a custom interaction loop for external execution environments. In this codebase, AppWorld is used as the environment backend, and communication between the trainer and AppWorld workers is implemented through Redis.

The two key implementation facts are:

1. The training stack is based on `verl`.
2. The backend interaction with AppWorld is mediated by Redis.

## System Design

The system is organized into two main layers:

- Training layer: `verl` handles rollout, optimization, batching, logging, and PPO-related components.
- Environment layer: AppWorld executes tool-using code and returns environment feedback.

Redis acts as the coordination layer between them:

- the trainer pushes execution requests into Redis queues,
- `app_worker.py` consumes those requests,
- workers execute code against AppWorld,
- execution outputs and evaluation signals are written back through Redis.

This design keeps the RL training process decoupled from the environment-serving process and makes multi-worker execution straightforward.

## Repository Layout

- `verl/`: PPO training infrastructure and algorithm implementations.
- `Tool_r1/`: custom generation and interaction logic for the agent loop.
- `app_worker.py`: Redis worker that initializes AppWorld tasks, executes code, and reports results.
- `startappworld.sh`: starts Redis, AppWorld environment servers, and worker processes.
- `train_start.sh`: main training entrypoint.
- `examples/grpo_trainer/train.sh`: trainer launch script built on `verl.trainer.main_ppo`.

## Training Flow

1. Start the AppWorld backend services.
2. Start training.
3. During rollout, the trainer sends execution requests through Redis.
4. AppWorld workers consume the requests and execute them.
5. Environment outputs are returned to the trainer and appended to the interaction history.
6. PPO optimization is performed on top of the resulting trajectories.

## Quick Start

For dependency setup, you can either:

- follow the installation instructions from the official `verl` repository, or
- install dependencies from this repository directly with `requirements.txt`.

Example:

```bash
pip install -r requirements.txt
```

`verl` and `AppWorld` should be treated as two different runtime environments.

- The `verl` environment is used for training.
- The `AppWorld` environment is used for the execution backend.

The AppWorld setup should follow the [official AppWorld repository](https://github.com/StonyBrookNLP/appworld).

The scripts should be launched in their corresponding environments:

- run the training script in the `verl` environment,
- run the backend startup script in the `AppWorld` environment.

Start the AppWorld backend:

```bash
bash startappworld.sh
```

Start training:

```bash
bash train_start.sh
```

## Configuration

The most commonly adjusted training settings are exposed in `train_start.sh`, including:

- model path
- dataset paths
- batch sizes
- prompt and response length limits
- PPO-related hyperparameters
- rollout configuration
- experiment and logging names

Backend startup settings are exposed in `startappworld.sh`, including:

- AppWorld root path
- Redis port
- Redis binary paths
- Python executable
- log directory

## Notes

- This repository assumes a local AppWorld deployment is available.
- Redis must be running for the trainer and workers to communicate.
- The project contains custom PPO variants and custom rollout logic on top of upstream `verl`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{dai2026group,
  title={Group Verification-based Policy Optimization for Interactive Coding Agents},
  author={Dai, Silong and Sun, Changzhi and Wu, Haolun and Zheng, Huanran and Ji, Tao and Yan, Junchi and Wu, Yuanbin and Zhang, Dell and Wang, Xiaoling and Li, Xuelong},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
