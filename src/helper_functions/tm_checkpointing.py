import os
import glob

def setup_checkpoint_dir(checkpoint_dir):
    """Create checkpoint directory if it doesn't exist."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def get_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint file."""
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    return None


def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3):
    """Remove old periodic checkpoints, keeping only the last N."""
    pattern = os.path.join(checkpoint_dir, "checkpoint_episode_*.pt")
    checkpoints = glob.glob(pattern)

    if len(checkpoints) > keep_last_n:
        checkpoints.sort(key=os.path.getctime)

        # Remove oldest checkpoints
        for checkpoint in checkpoints[:-keep_last_n]:
            try:
                os.remove(checkpoint)
                print(f"Removed old checkpoint: {checkpoint}")
            except Exception as e:
                print(f"Failed to remove checkpoint {checkpoint}: {e}")


def resume_from_checkpoint(agent, checkpoint_dir):
    """
    Attempt to resume training from a checkpoint.

    Args:
        agent: The agent object with a load_checkpoint method
        checkpoint_dir: Directory containing checkpoints

    Returns:
        tuple: (start_episode, start_step, wandb_run_id)
            - start_episode: Episode number to resume from (0 if no checkpoint)
            - start_step: Step number to resume from (0 if no checkpoint)
            - wandb_run_id: WandB run ID if found, None otherwise
    """
    start_episode = 0
    start_step = 0
    wandb_run_id = None

    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        try:
            checkpoint_data = agent.load_checkpoint(latest_checkpoint)
            start_episode = checkpoint_data['episode']
            start_step = checkpoint_data['step']
            # Get WandB run ID if it exists
            if 'additional_info' in checkpoint_data and 'run_id' in checkpoint_data['additional_info']:
                wandb_run_id = checkpoint_data['additional_info']['run_id']
                print(f"Resuming WandB run: {wandb_run_id}")
            print(f"Resuming training from episode {start_episode}, step {start_step}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting training from scratch")
    else:
        print("No checkpoint found, starting training from scratch")

    return start_episode, start_step, wandb_run_id