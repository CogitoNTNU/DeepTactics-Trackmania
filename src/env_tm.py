import gymnasium as gym
import os
import torch
import wandb
import glob
import time
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


def run_training():
    WANDB_API_KEY=os.getenv("WANDB_API_KEY")

    # Create IQN agent with optimal parameters
    dqn_agent = IQN()

    # Print device information
    print("="*50)
    print(f"Training on device: {dqn_agent.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*50)

    #env_name = "LunarLander-v3"
    env_name = "TM20"

    wandb.login(key=WANDB_API_KEY)

    # Configure video recording based on config
    print("setting envirometn")
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

    # Create descriptive run name
    run_name = f"IQN_Finally_on_TMRL"

    print("starting env")
    with wandb.init(project="Trackmania", name=run_name, config=dqn_agent.config) as run:
        run.watch(dqn_agent.policy_network, log="all", log_freq=100)
        run.watch(dqn_agent.target_network, log="all", log_freq=100)

        tot_reward = 0
        episode = 0
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
        print(type(observation))
        for i in range(tm_config.training_steps):
            
            obs_tensor = torch.tensor(observation[3][0], dtype=torch.float32)/255
            obs_tensor = obs_tensor.permute(2, 0, 1)
            action, q_value = dqn_agent.get_action(obs_tensor.unsqueeze(0))
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
            
            experience = TensorDict({
                "observation": obs_tensor,
                "action": torch.tensor(action),
                "reward": torch.tensor(reward),
                "next_observation": torch.tensor(next_obs[3][0], dtype=torch.float32).permute(2, 0, 1), # Next state
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
                    "steps": observation
                }

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

                observation, info = env.reset()
            else:
                observation = next_obs
                
            dqn_agent.decay_epsilon()

            if i % tm_config.target_network_update_frequency == 0:
                dqn_agent.update_target_network()

    env.close()

if __name__ == "__main__":
    run_training()


# print(observation[3].shape)
#             print(type(observation[0]),type(observation[1]),type(observation[2]))
#             print(type(observation[0][0]),type(observation[1][0]),type(observation[2][0]))
#             print(observation[0].shape,observation[1].shape,observation[2].shape)
#             print(type(observation[3][0]))
#             print(observation[3][0].shape)
