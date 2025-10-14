# Trackmania Infrastructure Setup

This document explains how to use the extracted linesight infrastructure in DeepTactics-Trackmania.

## What Was Extracted

We've created a **simplified single-process version** of linesight's Trackmania infrastructure:

### Files Added:
```
src/trackmania/
├── __init__.py
├── tminterface.py          # TMInterface socket communication
├── Python_Link.as          # AngelScript plugin for TMInterface
└── map_loader.py           # Map and zone center loading utilities

src/
├── trackmania_env.py       # Simplified Trackmania environment wrapper
└── train_trackmania.py     # Example training script

config_files/
└── tm_config.py            # Already existed, contains all settings
```

## Setup Instructions

### 1. Install TMInterface

1. Download TMInterface 2.x from https://donadigo.com/tminterface
2. Extract it to your Trackmania directory
3. Copy `src/trackmania/Python_Link.as` to your TMInterface Plugins folder:
   ```
   <Trackmania>/Plugins/Python_Link.as
   ```

### 2. Configure Paths

Edit `config_files/user_config.py` and set your Trackmania paths:

**Windows:**
```python
trackmania_base_path = Path("C:/Users/YourName/Documents/TrackMania")
windows_TMLoader_path = Path("C:/Path/To/TMLoader.exe")
windows_TMLoader_profile_name = "YourProfile"
```

**Linux:**
```python
trackmania_base_path = Path("/path/to/trackmania")
linux_launch_game_path = Path("/path/to/launch_script.sh")
```

### 3. Get Map Zone Centers

Zone centers are pre-computed racing lines saved as `.npy` files. You need these for training.

**Option A: Use linesight's maps**
Copy map files from linesight repo:
```bash
cp -r <linesight>/maps/ <deeptactics>/maps/
```

**Option B: Create your own**
See linesight documentation on generating zone centers from replays.

### 4. Modify IQN for Images

The current `src/IQN.py` only handles vector inputs (for LunarLander). You need to modify it to handle:
- **Images**: Game screen (160x120 grayscale)
- **Float features**: Speed, position, etc.

**Required changes to `Network` class:**

```python
class Network(nn.Module):
    def __init__(self, ...):
        super().__init__()

        # Add convolutional layers for images
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        # Add dense layers for float features
        self.float_fc1 = nn.Linear(float_input_dim, 128)
        self.float_fc2 = nn.Linear(128, 128)

        # Rest of your existing code...

    def forward(self, img: torch.Tensor, floats: torch.Tensor, n_tau: int = 8):
        # Process image
        x_img = F.relu(self.conv1(img))
        x_img = F.relu(self.conv2(x_img))
        x_img = F.relu(self.conv3(x_img))
        x_img = x_img.flatten(start_dim=1)

        # Process floats
        x_float = F.relu(self.float_fc1(floats))
        x_float = F.relu(self.float_fc2(x_float))

        # Merge
        x = torch.cat([x_img, x_float], dim=1)

        # Continue with your existing quantile embedding code...
```

**Reference:** See `linesight/trackmania_rl/agents/iqn.py` for a complete implementation.

## Running Training

### 1. Start Trackmania with TMInterface

1. Launch Trackmania
2. Load TMInterface
3. Make sure Python_Link plugin is loaded (check TMInterface console)

### 2. Run Training Script

```bash
python src/train_trackmania.py
```

The script will:
- Connect to TMInterface on port 8476 (configurable in `tm_config.py`)
- Load the specified map
- Run your IQN agent
- Log to Weights & Biases (optional)

## How It Works

### Environment Flow:

1. **Environment connects** to TMInterface via sockets
2. **Agent observes**: Gets game screenshot + simulation state
3. **Agent acts**: Sends inputs (accelerate, brake, left, right)
4. **Environment steps**: Game simulates forward
5. **Agent gets reward**: Based on progress through virtual checkpoints (zones)
6. Repeat until race finishes or timeout

### Key Simplifications from Linesight:

| Linesight (Complex) | DeepTactics (Simple) |
|---------------------|----------------------|
| Multi-process (collector + learner) | Single process |
| Multiple game instances | One game instance |
| Complex reward shaping | Simple distance-based reward |
| Advanced replay buffer | Basic replay buffer |
| Extensive logging & checkpointing | Minimal logging |

## Next Steps

1. **Test connection**: Run `src/train_trackmania.py` to verify TMInterface communication works
2. **Modify IQN**: Add conv layers for image processing
3. **Train**: Start with a simple map like A01-Race
4. **Iterate**: Add more sophisticated rewards, tune hyperparameters

## Troubleshooting

**"Connection refused"**
- Make sure Trackmania is running
- Check TMInterface is loaded
- Verify Python_Link.as is in Plugins folder
- Try restarting Trackmania

**"Map not found"**
- Check map path in `train_trackmania.py`
- Verify zone centers .npy file exists in `maps/` folder
- Check paths in `config_files/user_config.py`

**"Network input size mismatch"**
- You need to modify IQN to handle image inputs
- See "Modify IQN for Images" section above

## Differences from Linesight

This is a **learning-focused, single-process** implementation:

**Pros:**
- ✅ Much simpler to understand and debug
- ✅ Easier to modify and experiment with
- ✅ No multiprocessing complexity
- ✅ Good for prototyping algorithms

**Cons:**
- ❌ Slower training (one game instance vs many)
- ❌ Less optimized
- ❌ Missing some advanced features

**When to use linesight instead:**
- You need maximum training speed
- You're training on multiple maps
- You want all the bells and whistles

## Additional Resources

- TMInterface docs: https://donadigo.com/tminterface/documentation
- Linesight repo: https://github.com/pb4git/trackmania-rl
- IQN paper: https://arxiv.org/abs/1806.06923
