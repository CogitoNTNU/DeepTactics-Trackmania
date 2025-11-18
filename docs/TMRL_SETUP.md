# TrackMania Setup Guide

This guide explains how to set up TrackMania 2020 with the TMRL framework for training RL agents in the DeepTactics-TrackMania project.

> **Important Note**: This project uses **simplified configuration** compared to standard TMRL.
> **All training settings are controlled from `config_files/tm_config.py`** - you don't need to edit TMRL's `config.json` file.
> TMRL is used only as an environment interface to connect with TrackMania, not for its training pipeline.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Windows Installation](#windows-installation)
  - [Linux Installation](#linux-installation)
- [Configuration](#configuration)
- [Running Training](#running-training)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

---

## Prerequisites

### System Requirements

- **Operating System**: Windows 10/11 or Linux (TrackMania runs via Proton on Linux)
- **Python**: 3.13+ (already installed if you followed the main README)
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~10GB for TrackMania 2020 and dependencies

### Required Software

1. **TrackMania 2020** - The game itself
2. **OpenPlanet** - Community plugin for TrackMania (required for environment interface)
3. **TMRL Library** - Already included in project dependencies
4. **Virtual Gamepad Driver** - Automatically installed on Windows

---

## Installation

### Windows Installation

#### 1. Install TrackMania 2020

The free version of TrackMania 2020 is sufficient for this project.

1. Visit the [official TrackMania website](https://www.trackmania.com/)
2. Download and install TrackMania 2020
3. Create a Ubisoft account if you don't have one
4. Launch the game once to complete initial setup

#### 2. Install Visual C++ Runtime

OpenPlanet requires the Visual C++ runtime to function properly.

Download and install: [Visual C++ Runtime (64-bit)](https://aka.ms/vs/16/release/vc_redist.x64.exe)

#### 3. Install OpenPlanet

OpenPlanet is a community-supported utility that enables TMRL to interface with TrackMania.

1. Download from [OpenPlanet for TrackMania](https://openplanet.nl/)
2. Run the installer
3. **Important**: Windows may show a security warning (OpenPlanet is unsigned)
   - Click "More info"
   - Click "Install anyway"

#### 4. Verify TMRL Installation

The TMRL library should already be installed from the main project setup. Verify with:

```bash
python -c "import tmrl; print(tmrl.__version__)"
```

If not installed, install it:

```bash
uv add tmrl
```

#### 5. Install Virtual Gamepad Driver

When you run training for the first time, a virtual gamepad driver will be installed automatically. Accept the license agreement when prompted.

![Virtual Gamepad Driver](https://raw.githubusercontent.com/trackmania-rl/tmrl/master/readme/img/Nefarius1.png)

#### 6. Set Up TmrlData Folder

Navigate to your home folder (`C:\Users\your_username\`). You should find:

- **TmrlData** folder - Created by TMRL for configuration and checkpoints
- **OpenplanetNext** folder - Created by OpenPlanet (launch TrackMania once if missing)

**Verify OpenPlanet Plugin**:
1. Open `OpenplanetNext\Plugins`
2. Check that `TMRL_GrabData.op` exists
3. If missing, copy from `TmrlData\resources\Plugins` to `OpenplanetNext\Plugins`

---

### Linux Installation

Since Ubisoft does not officially support Linux, we use Proton to run TrackMania.

#### 1. Install TrackMania 2020 via Steam

1. Install Steam if not already installed
2. Enable Steam Play for all titles:
   - Steam → Settings → Compatibility
   - Check "Enable Steam Play for all other titles"
   - Select Proton version (e.g., Proton 8.0)

3. Install TrackMania 2020 from Steam
4. Launch the game once to complete setup

#### 2. Install OpenPlanet via Protontricks

```bash
# Install protontricks
sudo apt install protontricks  # Ubuntu/Debian
# or
sudo dnf install protontricks  # Fedora

# Find TrackMania's App ID
protontricks -l | grep -i trackmania

# Install OpenPlanet (replace APPID with actual ID)
protontricks APPID dlls vcrun2019
```

Download OpenPlanet installer and run via Proton prefix.

#### 3. Set Up uinput Permissions (Virtual Gamepad)

TMRL needs to create a virtual gamepad on Linux:

```bash
# Add user to input group
sudo usermod -a -G input $USER

# Create uinput rules
echo 'KERNEL=="uinput", MODE="0660", GROUP="input", OPTIONS+="static_node=uinput"' | \
  sudo tee /etc/udev/rules.d/99-uinput.rules

# Reload rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Load uinput module
sudo modprobe uinput

# Make it permanent
echo uinput | sudo tee /etc/modules-load.d/uinput.conf
```

Reboot for changes to take effect.

#### 4. Install xdotool (Window Control)

```bash
sudo apt install xdotool  # Ubuntu/Debian
# or
sudo dnf install xdotool  # Fedora
```

---

## Configuration

> **Note**: Unlike standard TMRL, you **do not need to configure `TmrlData/config/config.json`**.
> All training settings are in `config_files/tm_config.py`. The steps below cover only TrackMania game setup and verification.

### 1. Verify Installation

Launch TrackMania and verify the OpenPlanet plugin:

1. Launch TrackMania 2020
2. Press `F3` to open the OpenPlanet menu
3. Click `OpenPlanet > Log` to open logs
4. Go to `Developer > (Re)load plugin > TMRL Grab Data`
5. You should see "waiting for incoming connection" in the logs
6. Press `F3` again to close the menu

### 2. Game Settings

Configure TrackMania for optimal performance with TMRL:

**Graphics Settings**:
- **Resolution**: 1920x1080 (will be resized by TMRL)
- **Display Mode**: Windowed
- **Max FPS**: 30-60 (30 recommended for slower computers)
- **Graphics Quality**: Low to Medium (reduce if you see timeout warnings)

**Input Settings**:
- **Gamepad Input**: Leave at default (TMRL uses virtual gamepad)

### 3. Load Training Track

Our project uses custom tracks. To load them:

1. Copy track files (`.Map.Gbx`) from your project's `tracks/` folder
2. Paste into `Documents\Trackmania\Maps\My Maps\`
3. In TrackMania: `Create > Map Editor > Edit a Map > [your track] > Select Map`
4. Click the green flag to start

### 4. Camera Setup

**For Image-Based Training** (Rainbow agent):
- Press `1` key for default camera (car must be visible)
- Use the **Canadian flag skin** (recommended for consistency)

**For LIDAR Training** (if implemented):
- Press `3` key for cockpit view (car must be hidden)

### 5. Window Positioning

Position the TrackMania window:
- **Windowed mode**: Bring window to top-left corner
- On Windows 10/11, it should snap to a quarter of the screen
- Press `G` to hide the ghost car

---

## Running Training

> **Configuration**: All training settings are in `config_files/tm_config.py`.
> You control everything from this single file - no need to edit TMRL's `config.json`!

### 1. Configure Training Settings

Edit `config_files/tm_config.py` to customize your training:

```python
# Algorithm settings
use_dueling = True
use_prioritized_replay = True
crash_detection = True
crash_threshold = 10.0
crash_penalty = 10

# Network architecture
hidden_dim = 256
batch_size = 64
learning_rate = 0.0001

# Training hyperparameters
discount_factor = 0.997
n_step_buffer_len = 4
epsilon_decay_steps = 2_000_000

# Checkpoint settings
checkpoint = True
checkpoint_frequency = 10
resume_from_checkpoint = False  # Set to True to resume
```

### 2. Set Up WandB (Optional)

To track training metrics:

```bash
wandb login
# Enter your WandB API key
```

### 3. Start Training

In `main.py`, set:

```python
run_tm = True  # Enable TrackMania training
```

Then run:

```bash
uv run main.py
```

**Important**: After starting, quickly click inside the TrackMania window so TMRL can control the car.

### 4. Monitor Training

Watch the terminal for:
- Episode rewards
- Loss values
- Q-value estimates
- Epsilon decay
- **Warning**: "Timestep timeout" warnings (see Troubleshooting)

View detailed metrics on WandB dashboard (if enabled).

### 5. Checkpoints

Training checkpoints are saved to `checkpoints/`:
- `checkpoint_latest.pt` - Resume from here
- `checkpoint_episode_N.pt` - Periodic saves
- `replay_buffer.pkl` - Saved alongside checkpoints

---

## Troubleshooting

### Common Issues

#### "Connection refused" or "Communication error"

**Solution**:
1. Open TrackMania
2. Press `F3` → `Developer` → `(Re)load plugin` → `TMRL Grab Data`
3. Restart your training script

#### "DLL error from win32gui/win32ui/win32con"

**Solution**:
```bash
# Use conda instead of pip for pywin32
conda install pywin32
```

#### Many "timestep timeout" warnings

This means your computer struggles to run the AI and TrackMania in parallel.

**Solutions**:
- Reduce TrackMania graphics to minimum
- Set Max FPS to 30 in TrackMania settings
- Reduce `batch_size` in config (e.g., 32 instead of 64)
- Use smaller `hidden_dim` (e.g., 128 instead of 256)
- Consider remote training (train on HPC, run game locally)

#### Car not moving / controls not working

**Solutions**:
- Click inside the TrackMania window after starting training
- Verify virtual gamepad driver is installed (Windows)
- Check uinput permissions (Linux)
- Ensure Input settings are default in TrackMania

#### Crash detection too sensitive

**Solution**: Adjust in `config_files/tm_config.py`:
```python
crash_threshold = 15.0  # Increase threshold (default: 10.0)
crash_penalty = 5       # Reduce penalty (default: 10)
```

#### OpenPlanet plugin not found

**Solution**:
1. Navigate to `TmrlData\resources\`
2. Copy the `Plugins` folder
3. Paste into `OpenplanetNext\` folder
4. Reload plugin in TrackMania (F3 → Developer → Reload)

---

## Advanced Configuration

### Multi-Computer Training

You can run training across multiple computers (e.g., game on local PC, training on HPC).

This is **not currently implemented** in our project, but TMRL supports it via a server/worker/trainer architecture. See the [TMRL tutorial](https://github.com/trackmania-rl/tmrl/blob/master/readme/tuto_library.md) for details.

### Custom Reward Functions

To train on a new track with custom rewards:

1. Create/load your track in TrackMania
2. Record a reward function:
   ```bash
   python -m tmrl --record-reward
   ```
3. Complete the track (rewards computed automatically)
4. Verify rewards work:
   ```bash
   python -m tmrl --check-environment
   ```

**Note**: This requires integrating TMRL's CLI, which is not yet set up in our project.

### Environment Variations

TMRL supports different observation types:

**Full Environment** (Raw Screenshots):
- Input: RGB images (256x128 pixels)
- Downscaled to 64x64 for network input
- Uses Canadian flag car skin

**LIDAR Environment**:
- Input: Rangefinder beams from cockpit view
- Detects black pixels on road borders
- Requires tracks with plain roads (no decorations)

**LIDAR + Progress**:
- LIDAR + additional progress features
- Best for competition settings

Our project currently uses the **Full Environment** with image observations.

### Performance Benchmarking

To identify bottlenecks, TMRL provides benchmarking:

1. Set `"benchmark": true` in TMRL config
2. Run:
   ```bash
   python -m tmrl --benchmark
   ```
3. Analyze output:
   - `time_step_duration`: Time per step (should match target ~0.05s)
   - `inference_duration`: Policy forward pass time
   - `retrieve_obs_duration`: Screenshot capture time

---

## Differences from TMRL Defaults

Our project differs from standard TMRL setup:

| Aspect | TMRL Default | Our Project |
|--------|--------------|-------------|
| **Algorithm** | SAC (actor-critic) | Rainbow DQN (value-based) |
| **Training Mode** | Multi-process (server/worker/trainer) | Single-process |
| **Checkpointing** | TMRL's checkpoint system | Custom PyTorch checkpoints |
| **Configuration** | `config.json` in TmrlData | `config_files/tm_config.py` |
| **WandB Integration** | Built into TMRL CLI | Custom implementation in `env_tm.py` |
| **Replay Buffer** | TMRL's Memory class | Custom PER/standard buffer |

---

## Next Steps

After successful setup:

1. **Train on simple tracks** - Start with straight roads or gentle curves
2. **Monitor Q-values** - Should gradually increase (0 → 50+ over 2500 episodes)
3. **Adjust hyperparameters** - See `config_files/static_tm_config.py` for options
4. **Compare with Gymnasium** - Test your agent on CarRacing-v3 first
5. **Document findings** - Share insights with the team

For more details on TMRL, see the [official TMRL documentation](https://github.com/trackmania-rl/tmrl).

---

## References

- [TMRL GitHub Repository](https://github.com/trackmania-rl/tmrl)
- [TMRL Installation Guide](https://github.com/trackmania-rl/tmrl/blob/master/readme/Install.md)
- [TMRL Getting Started](https://github.com/trackmania-rl/tmrl/blob/master/readme/get_started.md)
- [TMRL Reference Guide](https://github.com/trackmania-rl/tmrl/blob/master/readme/reference_guide.md)
- [OpenPlanet Website](https://openplanet.nl/)
- [TrackMania Official Site](https://www.trackmania.com/)
