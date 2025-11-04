# DeepTactics-Trackmania AI Instructions

This project trains Deep Reinforcement Learning agents to play Trackmania using DQN variants, interfacing with the game via TMInterface 2.1.4.

## Architecture Overview

### Core Components
- **Entry Point**: `main.py` → `src/env.py:run_training()` 
- **RL Agent**: `src/DQN.py` - DQN with prioritized replay buffer and DDQN target updates
- **Game Interface**: `src/trackmania_interface/` - Complex TMInterface 2.1.4 socket communication layer
- **Configuration**: `config_files/` - Centralized config system with runtime parameter updates

### Training Loop Architecture
The project has dual training environments:
1. **Development/Testing**: CartPole-v1 environment (currently in `env.py`)
2. **Production**: Trackmania via `GameInstanceManager.rollout()` method

Key pattern: `GameInstanceManager.rollout(exploration_policy, map_path, zone_centers, update_network)` is the core training interface that:
- Manages game process lifecycle (launch/restart/close)
- Handles socket communication with TMInterface
- Collects visual frames and game state
- Executes agent policies and returns rollout results

## Development Workflows

### Environment Setup
```bash
# Install with uv (required)
uv sync
uv run pre-commit install  # Development only

# Run training
python main.py  # Currently uses CartPole, will need Trackmania setup

# Documentation
uv run mkdocs serve  # Local docs at http://127.0.0.1:8000/

# Testing
uv run pytest --doctest-modules --cov=src --cov-report=html
```

### TMInterface Integration
The Trackmania integration requires:
- TMInterface 2.1.4 installation and setup
- AngelScript plugin (`Python_Link.as`) for socket communication
- User configuration in `config_files/user_config.py` (Windows/Linux paths)

Critical: `src/trackmania_interface/game_instance_manager.py` is complex legacy code marked for refactoring - "If it works don't touch it." Prefer small, careful changes.

## Project-Specific Patterns

### Configuration System
- **Runtime Config Updates**: `tm_config.py` copied to `config_copy.py` during training for live parameter modification
- **Cross-Platform Support**: `tm_config.is_linux` switches between Windows/Linux game launch commands
- **Hyperparameters**: Extensive config in `tm_config.py` (epsilon schedules, network params, game settings)

### DQN Implementation Details
- **DDQN**: Uses policy network to select actions, target network for Q-value estimation
- **Prioritized Replay**: Optional PrioritizedReplayBuffer with importance sampling weights
- **Device Detection**: Auto-selects CUDA/MPS/CPU with proper tensor movement
- **Exploration**: Epsilon-greedy with configurable decay schedules

### Game Interface Patterns
- **Socket Communication**: Custom MessageType enum and binary protocol with TMInterface
- **Frame Processing**: Captures BGRA frames, converts to grayscale numpy arrays (H, W, 1)
- **Action Mapping**: Maps discrete action indices to Trackmania inputs (left/right/accelerate/brake)
- **Zone-Based Tracking**: Uses "zone centers" for progress tracking and reward calculation

### WandB Integration
```python
# Standard pattern for experiment tracking
with wandb.init(project="Trackmania") as run:
    run.watch(agent.policy_network, log="all", log_freq=100)
    # Log metrics: episode_reward, loss, epsilon, learning_rate, q_values
```

## Critical Files to Understand
- `src/trackmania_interface/game_instance_manager.py` - 769 lines of game integration logic
- `config_files/tm_config.py` - All hyperparameters and training configuration
- `src/DQN.py` - Core RL algorithm with prioritized replay and DDQN
- `src/trackmania_interface/tminterface2.py` - Socket protocol implementation

## Common Tasks
- **Add new RL algorithms**: Follow `DQN.py` pattern with `get_action()`, `train()`, `store_transition()` methods
- **Modify training**: Update `config_files/tm_config.py` for hyperparameters, `src/env.py` for training loop
- **Debug game connection**: Check TMInterface process, port configuration, and `user_config.py` paths
- **Performance tuning**: Monitor instrumentation metrics in GameInstanceManager rollout results

Remember: This project emphasizes group learning - "every line of code should be understood by the group."