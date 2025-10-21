# IQN Code Refactoring

The original `custom_actor_module.py` has been refactored into a more modular structure for better maintainability and clarity.

## New Structure

```
tmrl/
├── models/                     # Neural network models
│   ├── __init__.py
│   ├── iqn_network.py         # Base IQN CNN network
│   ├── iqn_actor.py           # Actor module (policy network)
│   ├── iqn_critic.py          # Critic module (Q-function)
│   └── iqn_actor_critic.py    # Combined actor-critic module
│
├── training/                   # Training components
│   ├── __init__.py
│   └── iqn_training_agent.py  # IQN training algorithm
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── nn_utils.py            # Neural network utilities (mlp, conv2d_out_dims, etc.)
│   └── serialization.py       # JSON serialization for PyTorch models
│
├── config/                     # Configuration
│   └── iqn_config.py          # All configuration parameters
│
└── run_iqn.py                 # Main entry point
```

## File Descriptions

### Models (`tmrl/models/`)

- **`iqn_network.py`**: Base IQN CNN network that implements the core distributional RL architecture using quantile regression
- **`iqn_actor.py`**: Actor module (`MyActorModule`) that implements the policy network for action selection
- **`iqn_critic.py`**: Critic module (`IQNCNNQFunction`) that estimates action-values using IQN
- **`iqn_actor_critic.py`**: Combines actor and dual critics for the full actor-critic architecture

### Training (`tmrl/training/`)

- **`iqn_training_agent.py`**: Implements the IQN training algorithm with quantile Huber loss and actor-critic updates

### Utils (`tmrl/utils/`)

- **`nn_utils.py`**: Neural network utility functions (MLP builder, feature counting, conv2d dimension calculation)
- **`serialization.py`**: JSON encoders/decoders for safe model serialization (competition requirement)

### Configuration (`tmrl/config/`)

- **`iqn_config.py`**: All training parameters, hyperparameters, and configuration setup

### Entry Point

- **`run_iqn.py`**: Main script that launches server, trainer, or worker based on command-line arguments

## Usage

The refactored code works exactly the same as before:

```bash
# Launch server
python tmrl/run_iqn.py --server

# Launch trainer
python tmrl/run_iqn.py --trainer

# Launch worker
python tmrl/run_iqn.py --worker

# Or use the batch files (already updated):
run_iqn_server.bat
run_iqn_trainer.bat
run_iqn_worker.bat
run_iqn_all.bat
```

## Benefits of Refactoring

1. **Modularity**: Each class has its own file, making it easier to find and modify specific components
2. **Maintainability**: Changes to one component (e.g., actor) don't require scrolling through 1000+ lines
3. **Readability**: Clear separation of concerns with organized directory structure
4. **Reusability**: Individual components can be imported and reused in other projects
5. **Testing**: Easier to write unit tests for individual components
6. **Collaboration**: Multiple developers can work on different components simultaneously

## Backward Compatibility

The original `custom_actor_module.py` is still present and functional. The refactored version is a drop-in replacement with identical functionality.
