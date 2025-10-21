"""
Main entry point for running IQN-based TrackMania RL training.

This script can launch the server, trainer, or worker components of the
distributed training pipeline.

Usage:
    python run_iqn.py --server   # Launch the central server
    python run_iqn.py --trainer  # Launch the trainer
    python run_iqn.py --worker   # Launch a rollout worker
    python run_iqn.py --test     # Launch a standalone test worker
"""

import sys
import os
from pathlib import Path

# Add the tmrl directory to Python path so local modules can be imported
tmrl_dir = Path(__file__).parent
if str(tmrl_dir) not in sys.path:
    sys.path.insert(0, str(tmrl_dir))

from argparse import ArgumentParser
from tmrl.util import partial
from tmrl.networking import Trainer, RolloutWorker, Server

# Import local modules
from config.iqn_config import (
    training_cls,
    env_cls,
    sample_compressor,
    obs_preprocessor,
    server_ip_for_trainer,
    server_ip_for_worker,
    server_port,
    password,
    security,
    device_worker,
    max_samples_per_episode,
    wandb_entity,
    wandb_project,
    wandb_run_id
)

from models.iqn_actor import MyActorModule
from training.iqn_training_agent import IQNTrainingAgent
from models.iqn_actor_critic import IQNCNNActorCritic
from models.iqn_network import IQNCNN
from models.iqn_critic import IQNCNNQFunction

# IMPORTANT: For pickle backward compatibility
# When checkpoints are loaded, Python looks for classes in __main__
# So we need to expose our classes here for old checkpoints to work
__all__ = [
    'MyActorModule',
    'IQNTrainingAgent',
    'IQNCNNActorCritic',
    'IQNCNN',
    'IQNCNNQFunction'
]


def main():
    """Main function to parse arguments and launch the appropriate component."""
    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true', help='launches the server')
    parser.add_argument('--trainer', action='store_true', help='launches the trainer')
    parser.add_argument('--worker', action='store_true', help='launches a rollout worker')
    parser.add_argument('--test', action='store_true', help='launches a rollout worker in standalone mode')
    args = parser.parse_args()

    if args.trainer:
        print("Starting IQN Trainer with W&B logging...")
        print(f"W&B Project: {wandb_project}, Run ID: {wandb_run_id}")
        my_trainer = Trainer(training_cls=training_cls,
                             server_ip=server_ip_for_trainer,
                             server_port=server_port,
                             password=password,
                             security=security)

        # Enable wandb logging (uses your existing wandb login)
        my_trainer.run_with_wandb(entity=wandb_entity,
                                  project=wandb_project,
                                  run_id=wandb_run_id)

    elif args.worker or args.test:
        print(f"Starting IQN Worker (test mode: {args.test})...")
        # Partially instantiate the actor module with the n_quantiles parameter
        actor_module_cls = partial(MyActorModule, n_quantiles=64)

        rw = RolloutWorker(env_cls=env_cls,
                           actor_module_cls=actor_module_cls,
                           sample_compressor=sample_compressor,
                           device=device_worker,
                           server_ip=server_ip_for_worker,
                           server_port=server_port,
                           password=password,
                           security=security,
                           max_samples_per_episode=max_samples_per_episode,
                           obs_preprocessor=obs_preprocessor,
                           standalone=args.test)
        rw.run(test_episode_interval=10)

    elif args.server:
        print("Starting IQN Server...")
        import time
        serv = Server(port=server_port,
                      password=password,
                      security=security)
        while True:
            time.sleep(1.0)

    else:
        print("Please specify one of: --server, --trainer, --worker, or --test")
        parser.print_help()


if __name__ == "__main__":
    main()
