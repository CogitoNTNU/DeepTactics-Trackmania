import gymnasium as gym
import os
import torch
import wandb
import glob
import time
from pathlib import Path
from src.IQN_cnn import IQN
from tensordict import TensorDict
from gymnasium.wrappers import RecordVideo
from config_files import tm_config

from tmrl import get_environment
from time import sleep
import numpy as np


def map_action_tm(idx):
                # Steering in [-1, 1], accel/brake in [0, 1]
                # steering -1 is left steering 1 is right, can be f.ex -0.3.
                # Adjust/add combos as you need and keep this consistent with your agent's action space.
                mapping = {
                    0: np.array([0.0, 0.0, 0.0], dtype=np.float32),   # no-op / coast
                    1: np.array([1.0, 0.0, 0.0], dtype=np.float32),   # accelerate
                    2: np.array([1.0, 1.0, 0.0], dtype=np.float32),   # brake and accelerate
                    3: np.array([0.0, 1.0, 0.0], dtype=np.float32),   # brake
                    4: np.array([0.0, 0.5, 0.0], dtype=np.float32),   # half brake
                    5: np.array([1.0, 0.0, -1.0], dtype=np.float32),  # left + light accel
                    6: np.array([1.0, 0.0, 1.0], dtype=np.float32),   # right + light accel
                    7: np.array([0.0, 1.0, -1.0], dtype=np.float32),  # left + brake
                    8: np.array([0.0, 1.0, 1.0], dtype=np.float32),   # right + brake
                    9: np.array([0.0, 0.0, -1.0], dtype=np.float32),  # left
                    10: np.array([0.0, 0.0, 1.0], dtype=np.float32),  # right
                    11: np.array([0.0, 0.0, -0.3], dtype=np.float32), # slight left
                    12: np.array([0.0, 0.0, 0.3], dtype=np.float32),  # slight right
                }
                return mapping.get(idx, mapping[0])


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
        # Sort by creation time
        checkpoints.sort(key=os.path.getctime)
        # Remove oldest checkpoints
        for checkpoint in checkpoints[:-keep_last_n]:
            try:
                os.remove(checkpoint)
                print(f"Removed old checkpoint: {checkpoint}")
            except Exception as e:
                print(f"Failed to remove checkpoint {checkpoint}: {e}")


def run_training():
    WANDB_API_KEY=os.getenv("WANDB_API_KEY")

    # Create descriptive run name
    run_name = f"IQN_Finally_on_TMRL"

    # Setup checkpoint directory with run name
    checkpoint_dir = os.path.join(tm_config.checkpoint_dir, run_name)
    checkpoint_dir = setup_checkpoint_dir(checkpoint_dir)
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Create IQN agent with optimal parameters
    dqn_agent = IQN()

    # Print device information
    print("="*50)
    print(f"Training on device: {dqn_agent.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*50)

    # Try to load from checkpoint if resume is enabled
    start_episode = 0
    start_step = 0
    wandb_run_id = None
    if tm_config.resume_from_checkpoint:
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            try:
                checkpoint_data = dqn_agent.load_checkpoint(latest_checkpoint)
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

    #env_name = "LunarLander-v3"
    env_name = "TM20"

    wandb.login(key=WANDB_API_KEY)

    # Configure video recording based on config
    print("setting environment")
    if env_name == "TM20":
        env = get_environment()
        sleep(1.0)
        print("Enviroment set as TM20")
    else:
        if tm_config.record_video:
            #env = gym.make(env_name, render_mode="rgb_array")
            env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
            episode_record_frequency = 20
            video_folder = f"{env_name}-training"
            env = RecordVideo(
                env,
                video_folder=video_folder,
                name_prefix="eval",
                episode_trigger=lambda x: x % episode_record_frequency == 0,
            )
        else:
            env = gym.make("CarRacing-v3", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
            #env = gym.make(env_name)  # No rendering for faster training
            video_folder = None

    print("starting env")
    # Resume WandB run if we have a run_id, otherwise create new
    if wandb_run_id:
        print(f"Resuming existing WandB run: {wandb_run_id}")
        run_context = wandb.init(project="Trackmania", name=run_name, id=wandb_run_id, resume="allow", config=dqn_agent.config)
    else:
        print("Creating new WandB run")
        run_context = wandb.init(project="Trackmania", name=run_name, config=dqn_agent.config)

    with run_context as run:
        run.watch(dqn_agent.policy_network, log="all", log_freq=100)
        run.watch(dqn_agent.target_network, log="all", log_freq=100)

        tot_reward = 0
        episode = start_episode
        tot_q_value = 0
        n_q_values = 0

        #for TM20FULL obs = [velocity, gear, rpm, images]
        #images greyscale obs[3].shape >>> (IMG_HIST_LEN,x,y)
        #images full color obs[3].shape >>> (IMG_HIST_LEN,x,y,rgb(3))
        #velocity type(obs[0]) >>> array(1,) || velocity type(obs[0][0]) >> numpy.float32
        #gear type(obs[1]) >>> array(1,) || gear type(obs[1][0]) >> numpy.float32
        #rpm type(obs[2]) >>> array(1,) || rpm type(obs[2][0]) >> numpy.float32
        #example obs:[001.4, 0.0, 01772.1, imgs(1)] obs:[028.0, 2.0, 06113.0, imgs(1)]

        observation, _ = env.reset()

        try:
            for i in range(start_step, tm_config.training_steps):

                image_tensor = torch.tensor(observation[3], dtype=torch.float32)/255
                # image_tensor = torch.tensor(observation[3][0], dtype=torch.float32)/255
                # image_tensor = image_tensor.unsqueeze(0)
                # image_tensor = image_tensor.permute(2, 0, 1) # for color images
                car_features = torch.tensor([observation[0][0], observation[1][0], observation[2][0]], dtype=torch.float32).unsqueeze(0)
                action, q_value = dqn_agent.get_action(image_tensor.unsqueeze(0),car_features.unsqueeze(0))
                # print(f"Action: {action}")
                # Ensure action is a plain int (agent might return a tensor)
                if hasattr(action, "item"):
                    action_idx = int(action.item())
                else:
                    action_idx = int(action)

                # Map discrete action index -> Trackmania control vector [steer, accel, brake]
                mapped_action = map_action_tm(action_idx)

                if q_value is not None:
                    tot_q_value += q_value
                    n_q_values += 1

                next_obs, reward, terminated, truncated, _ = env.step(mapped_action)
                done = terminated or truncated
                
                # Process next observation tensors
                next_image_tensor = torch.tensor(next_obs[3], dtype=torch.float32)/255 #for batched images remember to update conv batch size
                # next_image_tensor = torch.tensor(next_obs[3][0], dtype=torch.float32) / 255 #for singel image remember to update conv batch size
                # next_image_tensor = next_image_tensor.unsqueeze(0) #for singel image
                # next_image_tensor = next_image_tensor.permute(2, 0, 1) # for singel color image
                next_car_features = torch.tensor([next_obs[0][0], next_obs[1][0], next_obs[2][0]], dtype=torch.float32)
                
                experience = TensorDict({
                    "image": image_tensor,
                    "car_features": car_features,
                    "action": torch.tensor(action),
                    "reward": torch.tensor(reward),
                    "next_image": next_image_tensor,
                    "next_car_features": next_car_features,
                    "done": torch.tensor(done)
                }, batch_size=torch.Size([]))

                dqn_agent.store_transition(experience)
                tot_reward += float(reward)

                loss = dqn_agent.train()

                if done:
                    if n_q_values > 0:
                        avg_q_value = tot_q_value / n_q_values
                    else:
                        avg_q_value = -1

                    log_metrics = {
                        "episode_reward": tot_reward,
                        "loss": loss,
                        "learning_rate": dqn_agent.optimizer.param_groups[0]['lr'],
                        "q_values": avg_q_value,
                        "epsilon": dqn_agent.epsilon,
                    }

                    dqn_agent.decay_epsilon()

                    # Only process videos if recording is enabled
                    if tm_config.record_video and video_folder:
                        video_path = None
                        pattern = os.path.join(video_folder, "*.mp4")
                        deadline = time.time() + 2
                        while time.time() < deadline:
                            candidates = glob.glob(pattern)
                            if candidates:
                                video_path = max(candidates, key=os.path.getctime)
                                break

                        if video_path:
                            log_metrics["episode_video"] = wandb.Video(video_path, format="mp4", caption=f"Episode {episode}")

                    run.log(log_metrics, step=episode)

                    episode += 1
                    tot_reward = 0
                    tot_q_value = 0
                    n_q_values = 0

                    # Save checkpoint every N episodes
                    if episode % tm_config.checkpoint_frequency == 0:
                        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode}.pt")
                        latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")

                        dqn_agent.save_checkpoint(checkpoint_path, episode, i, {"run_id": run.id})
                        dqn_agent.save_checkpoint(latest_path, episode, i, {"run_id": run.id})

                        # Clean up old checkpoints
                        cleanup_old_checkpoints(checkpoint_dir, tm_config.keep_last_n_checkpoints)

                    observation, _ = env.reset()
                else:
                    observation = next_obs

                if i % tm_config.target_network_update_frequency == 0:
                    dqn_agent.update_target_network()

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            crash_path = os.path.join(checkpoint_dir, "checkpoint_interrupted.pt")
            print(f"Saving checkpoint due to interruption...")
            dqn_agent.save_checkpoint(crash_path, episode, i, {"run_id": run.id, "reason": "user_interrupt"})
            raise

        except Exception as e:
            print(f"\nTraining crashed with error: {e}")
            crash_path = os.path.join(checkpoint_dir, "checkpoint_crash.pt")
            print(f"Saving emergency checkpoint...")
            try:
                dqn_agent.save_checkpoint(crash_path, episode, i, {"run_id": run.id, "error": str(e)})
            except Exception as save_error:
                print(f"Failed to save crash checkpoint: {save_error}")
            raise

        finally:
            # Always save a final checkpoint when training ends
            final_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
            print(f"Saving final checkpoint...")
            try:
                dqn_agent.save_checkpoint(final_path, episode, i, {"run_id": run.id})
            except Exception as save_error:
                print(f"Failed to save final checkpoint: {save_error}")

    env.close()

if __name__ == "__main__":
    run_training()


# print(observation[3].shape)
#             print(type(observation[0]),type(observation[1]),type(observation[2]))
#             print(type(observation[0][0]),type(observation[1][0]),type(observation[2][0]))
#             print(observation[0].shape,observation[1].shape,observation[2].shape)
#             print(type(observation[3][0]))
#             print(observation[3][0].shape)
