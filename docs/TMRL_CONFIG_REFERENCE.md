# TMRL Configuration Reference

This document provides a quick reference for TMRL configuration parameters that may be useful when extending the DeepTactics-TrackMania project to use TMRL's full capabilities.

> **Note**: Our current implementation uses a simplified single-process training setup with custom configuration in `config_files/tm_config.py`. The settings below are for reference if you want to use TMRL's standard multi-process architecture (Server/Trainer/Worker).

---

## Configuration File Location

TMRL uses a `config.json` file located at:
- **Windows**: `C:\Users\username\TmrlData\config\config.json`
- **Linux**: `~/TmrlData/config/config.json`

---

## Key Configuration Parameters

### Run Configuration

```json
{
  "RUN_NAME": "SAC_4_imgs_pretrained",
  "RESET_TRAINING": false,
  "BUFFERS_MAXLEN": 500000,
  "RW_MAX_SAMPLES_PER_EPISODE": 1000
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RUN_NAME` | `"SAC_4_imgs_pretrained"` | Experiment name (used for checkpoint/weight file naming) |
| `RESET_TRAINING` | `false` | If true, restart training from scratch (keeps replay buffer) |
| `BUFFERS_MAXLEN` | `500000` | Maximum length of local buffers (NOT the replay buffer) |
| `RW_MAX_SAMPLES_PER_EPISODE` | `1000` | Force episode truncation if longer than this |

### Hardware Configuration

```json
{
  "CUDA_TRAINING": true,
  "CUDA_INFERENCE": false,
  "VIRTUAL_GAMEPAD": true
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CUDA_TRAINING` | `true` | Use GPU for training (Trainer process) |
| `CUDA_INFERENCE` | `false` | Use GPU for inference (Worker process) |
| `VIRTUAL_GAMEPAD` | `true` | Use virtual gamepad for TrackMania control |

### Network Configuration

```json
{
  "LOCALHOST_WORKER": true,
  "LOCALHOST_TRAINER": true,
  "PUBLIC_IP_SERVER": "0.0.0.0",
  "PORT": 55555,
  "LOCAL_PORT_SERVER": 55556,
  "LOCAL_PORT_TRAINER": 55557,
  "LOCAL_PORT_WORKER": 55558
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LOCALHOST_WORKER` | `true` | Set to `false` if Worker is not on same machine as Server |
| `LOCALHOST_TRAINER` | `true` | Set to `false` if Trainer is not on same machine as Server |
| `PUBLIC_IP_SERVER` | `"0.0.0.0"` | Server IP address when not on localhost |
| `PORT` | `55555` | Public port of Server (must be forwarded if remote) |
| `LOCAL_PORT_SERVER` | `55556` | Localhost Server port (must not overlap) |
| `LOCAL_PORT_TRAINER` | `55557` | Localhost Trainer port (must not overlap) |
| `LOCAL_PORT_WORKER` | `55558` | **Change this if multiple Workers on same machine!** |

### Security Configuration

```json
{
  "PASSWORD": "YourRandomPasswordHere",
  "TLS": false,
  "TLS_HOSTNAME": "default",
  "TLS_CREDENTIALS_DIRECTORY": "",
  "NB_WORKERS": -1
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PASSWORD` | `"YourRandomPasswordHere"` | Must match on all machines (Server/Trainer/Worker) |
| `TLS` | `false` | **Set to `true` when using TMRL on public networks!** |
| `TLS_HOSTNAME` | `"default"` | TLS hostname (for custom tlspyo configuration) |
| `TLS_CREDENTIALS_DIRECTORY` | `""` | TLS credential directory (for custom tlspyo config) |
| `NB_WORKERS` | `-1` | Maximum number of Workers that can connect (-1 = infinite) |

### Weights & Biases Integration

```json
{
  "WANDB_PROJECT": "tmrl",
  "WANDB_ENTITY": "tmrl",
  "WANDB_KEY": "YourWandbApiKey"
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WANDB_PROJECT` | `"tmrl"` | Your WandB project name |
| `WANDB_ENTITY` | `"tmrl"` | Your WandB entity name |
| `WANDB_KEY` | `"YourWandbApiKey"` | Your WandB API key (get from wandb.ai) |

### Training Hyperparameters

```json
{
  "MAX_EPOCHS": 10000,
  "ROUNDS_PER_EPOCH": 100,
  "TRAINING_STEPS_PER_ROUND": 200,
  "MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP": 4.0,
  "ENVIRONMENT_STEPS_BEFORE_TRAINING": 1000,
  "UPDATE_MODEL_INTERVAL": 200,
  "UPDATE_BUFFER_INTERVAL": 200,
  "SAVE_MODEL_EVERY": 0,
  "MEMORY_SIZE": 1000000,
  "BATCH_SIZE": 256
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_EPOCHS` | `10000` | Maximum number of training "epochs" (checkpoint/wandb after each) |
| `ROUNDS_PER_EPOCH` | `100` | Training "rounds" per epoch (metrics displayed after each) |
| `TRAINING_STEPS_PER_ROUND` | `200` | Number of training iterations per "round" |
| `MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP` | `4.0` | Pause training if ratio exceeds this |
| `ENVIRONMENT_STEPS_BEFORE_TRAINING` | `1000` | Minimum samples before training starts |
| `UPDATE_MODEL_INTERVAL` | `200` | Model updates at this interval (training steps) |
| `UPDATE_BUFFER_INTERVAL` | `200` | Sample retrieval interval (training steps) |
| `SAVE_MODEL_EVERY` | `0` | Save model copy at this interval (0 = no history) |
| `MEMORY_SIZE` | `1000000` | **Replay buffer size** (max samples in memory) |
| `BATCH_SIZE` | `256` | Training batch size |

### Algorithm Configuration (SAC)

```json
{
  "ALG": {
    "ALGORITHM": "SAC",
    "LEARN_ENTROPY_COEF": false,
    "LR_ACTOR": 0.00001,
    "LR_CRITIC": 0.00005,
    "LR_ENTROPY": 0.0003,
    "GAMMA": 0.995,
    "POLYAK": 0.995,
    "TARGET_ENTROPY": -0.5,
    "ALPHA": 0.01,
    "OPTIMIZER_ACTOR": "adam",
    "OPTIMIZER_CRITIC": "adam",
    "BETAS_ACTOR": [0.997, 0.997],
    "BETAS_CRITIC": [0.997, 0.997],
    "L2_ACTOR": 0.0,
    "L2_CRITIC": 0.0
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ALGORITHM` | `"SAC"` | Algorithm name (`"SAC"` or `"REDQSAC"`) |
| `LEARN_ENTROPY_COEF` | `false` | `true` for SAC v2 (learnable entropy coefficient) |
| `LR_ACTOR` | `0.00001` | Actor learning rate |
| `LR_CRITIC` | `0.00005` | Critic learning rate |
| `LR_ENTROPY` | `0.0003` | Entropy coefficient learning rate (SAC v2) |
| `GAMMA` | `0.995` | Discount factor |
| `POLYAK` | `0.995` | Polyak averaging factor for target critic |
| `TARGET_ENTROPY` | `-0.5` | Target entropy (SAC v2) |
| `ALPHA` | `0.01` | Entropy coefficient (constant for SAC, initial for SAC v2) |
| `OPTIMIZER_ACTOR` | `"adam"` | Actor optimizer (`"adam"`, `"adamw"`, `"sgd"`) |
| `OPTIMIZER_CRITIC` | `"adam"` | Critic optimizer (`"adam"`, `"adamw"`, `"sgd"`) |
| `BETAS_ACTOR` | `[0.997, 0.997]` | Actor Adam/AdamW betas |
| `BETAS_CRITIC` | `[0.997, 0.997]` | Critic Adam/AdamW betas |
| `L2_ACTOR` | `0.0` | Actor weight decay (Adam/AdamW) |
| `L2_CRITIC` | `0.0` | Critic weight decay (Adam/AdamW) |

### REDQ-SAC Specific

```json
{
  "ALG": {
    "ALGORITHM": "REDQSAC",
    "REDQ_N": 10,
    "REDQ_M": 2,
    "REDQ_Q_UPDATES_PER_POLICY_UPDATE": 20
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `REDQ_N` | `10` | Number of critic networks (ensemble size) |
| `REDQ_M` | `2` | Random subset size for Q-value calculation |
| `REDQ_Q_UPDATES_PER_POLICY_UPDATE` | `20` | Critic updates per policy update |

### Environment Configuration

```json
{
  "ENV": {
    "RTGYM_INTERFACE": "TM20FULL",
    "WINDOW_WIDTH": 256,
    "WINDOW_HEIGHT": 128,
    "IMG_WIDTH": 64,
    "IMG_HEIGHT": 64,
    "IMG_GRAYSCALE": true,
    "SLEEP_TIME_AT_RESET": 1.5,
    "IMG_HIST_LEN": 4
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RTGYM_INTERFACE` | `"TM20FULL"` | Environment type (`"TM20FULL"`, `"TM20LIDAR"`, `"TM20LIDARPROGRESS"`) |
| `WINDOW_WIDTH` | `256` | TrackMania window width (pixels) |
| `WINDOW_HEIGHT` | `128` | TrackMania window height (pixels) |
| `IMG_WIDTH` | `64` | Screenshot width for model input |
| `IMG_HEIGHT` | `64` | Screenshot height for model input |
| `IMG_GRAYSCALE` | `true` | Convert images to grayscale |
| `SLEEP_TIME_AT_RESET` | `1.5` | Wait time after respawn (for green light) |
| `IMG_HIST_LEN` | `4` | Number of frames in observation history |

### RTGym Configuration

```json
{
  "RTGYM_CONFIG": {
    "time_step_duration": 0.05,
    "start_obs_capture": 0.04,
    "time_step_timeout_factor": 1.0,
    "act_buf_len": 2,
    "benchmark": false,
    "wait_on_done": true,
    "ep_max_length": 1000,
    "interface_kwargs": {
      "save_replays": false
    }
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `time_step_duration` | `0.05` | Target time-step duration (seconds) - 20 FPS |
| `start_obs_capture` | `0.04` | Start observation capture after this time (seconds) |
| `time_step_timeout_factor` | `1.0` | Maximum elasticity for time-step duration |
| `act_buf_len` | `2` | Number of actions in action history |
| `benchmark` | `false` | Set to `true` for `--benchmark` command |
| `wait_on_done` | `true` | Used in TrackMania pipeline |
| `ep_max_length` | `1000` | Maximum episode length (truncation) |
| `save_replays` | `false` | Save TrackMania replays (for videos) |

### Reward Configuration

```json
{
  "REWARD_CONFIG": {
    "END_OF_TRACK": 100.0,
    "CONSTANT_PENALTY": 0.0,
    "CHECK_FORWARD": 500,
    "CHECK_BACKWARD": 10,
    "FAILURE_COUNTDOWN": 10,
    "MIN_STEPS": 70,
    "MAX_STRAY": 100.0
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `END_OF_TRACK` | `100.0` | Reward for crossing finish line |
| `CONSTANT_PENALTY` | `0.0` | Constant reward per step (usually 0 or negative) |
| `CHECK_FORWARD` | `500` | Look-ahead distance for progress (longer = more computation, enables cuts) |
| `CHECK_BACKWARD` | `10` | Look-back distance (10 is fine, longer = more reliable) |
| `FAILURE_COUNTDOWN` | `10` | Episode terminates after this many steps with no reward |
| `MIN_STEPS` | `70` | Initial steps before episode can terminate |
| `MAX_STRAY` | `100.0` | Episode terminates if car wanders further than this (meters) |

---

## Environment Types

### TM20FULL (Raw Screenshots)

**Observations**: RGB or grayscale images (default: 64x64 grayscale)
**Best for**: General tracks, visual feature learning
**Requirements**:
- Camera: Default view (press `1` key, car visible)
- Skin: Canadian flag (recommended for consistency)
- Window: 256x128 pixels

**Example Config**:
```json
{
  "ENV": {
    "RTGYM_INTERFACE": "TM20FULL",
    "WINDOW_WIDTH": 256,
    "WINDOW_HEIGHT": 128,
    "IMG_WIDTH": 64,
    "IMG_HEIGHT": 64,
    "IMG_GRAYSCALE": true,
    "IMG_HIST_LEN": 4
  }
}
```

### TM20LIDAR

**Observations**: Rangefinder beams detecting road borders
**Best for**: Tracks with plain roads (no decorations)
**Requirements**:
- Camera: Cockpit view (press `3` key, car hidden)
- Track: Must have black pixels on road borders
- Window: 958x488 pixels

**Example Config**:
```json
{
  "ENV": {
    "RTGYM_INTERFACE": "TM20LIDAR",
    "WINDOW_WIDTH": 958,
    "WINDOW_HEIGHT": 488,
    "IMG_HIST_LEN": 4
  }
}
```

### TM20LIDARPROGRESS

**Observations**: LIDAR + additional progress features
**Best for**: Competition settings
**Requirements**: Same as TM20LIDAR

---

## TMRL Command Line Interface

### Training Commands

```bash
# Launch server (coordinates communication)
python -m tmrl --server

# Launch trainer (runs training algorithm)
python -m tmrl --trainer

# Launch trainer with WandB logging
python -m tmrl --trainer --wandb

# Launch worker (collects samples from environment)
python -m tmrl --worker

# Launch test worker (no training sample collection)
python -m tmrl --test
```

### Setup Commands

```bash
# Install/regenerate TmrlData folder
python -m tmrl --install

# Record reward function for a track
python -m tmrl --record-reward

# Check environment and reward sanity
python -m tmrl --check-environment

# Benchmark environment performance
python -m tmrl --benchmark
```

### Advanced Commands

```bash
# Launch expert worker (ignores model updates)
python -m tmrl --expert

# Override config.json at runtime
python -m tmrl --trainer --config='{"CUDA_TRAINING": True, "BATCH_SIZE": 128}'
```

---

## Adapting to Our Project

Our project uses **Rainbow DQN** instead of TMRL's default **SAC**, so the algorithm configuration differs:

| TMRL (SAC) | Our Project (Rainbow DQN) |
|------------|---------------------------|
| Actor-critic architecture | Value-based (Q-network only) |
| Continuous action space | Discrete action space (11 actions) |
| Policy gradient updates | Q-learning with target networks |
| Multi-process (server/trainer/worker) | Single-process training |
| `config.json` in TmrlData | `tm_config.py` in config_files |

To use TMRL's configuration with our Rainbow agent, you would need to:

1. Implement a server/worker/trainer architecture (currently single-process)
2. Adapt TMRL's environment wrapper to work with discrete actions
3. Create a config translator between `config.json` and `tm_config.py`

This is **future work** if you want to leverage TMRL's distributed training capabilities.

---

## References

- [TMRL Reference Guide](https://github.com/trackmania-rl/tmrl/blob/master/readme/reference_guide.md)
- [TMRL Installation Guide](https://github.com/trackmania-rl/tmrl/blob/master/readme/Install.md)
- [TMRL Python API Tutorial](https://github.com/trackmania-rl/tmrl/blob/master/readme/tuto_library.md)
- [TMRL Documentation](https://tmrl.readthedocs.io/)
