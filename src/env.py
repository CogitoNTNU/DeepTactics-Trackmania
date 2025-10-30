# env.py

import os
import glob
import time
import numpy as np
import torch
import wandb
import gymnasium as gym
from tensordict import TensorDict
from gymnasium.wrappers import ClipAction, RecordVideo, TransformObservation
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit

from src.IQN import IQN
from config_files import tm_config


ant_kwargs = {
    "xml_file": "ant.xml",
    "forward_reward_weight": 1.0,
    "ctrl_cost_weight": 0.5,
    "contact_cost_weight": 5e-4,
    "healthy_reward": 1.0,
    "main_body": 1,
    "terminate_when_unhealthy": True,
    "healthy_z_range": (0.3, 1.0),
    "contact_force_range": (-1.0, 1.0),
    "reset_noise_scale": 0.1,
    "exclude_current_positions_from_observation": True,
    "include_cfrc_ext_in_observation": True,
}

eps_start, eps_end, eps_decay = 1.0, 0.05, 2e4  # decay over 200k steps


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env: gym.Env, action_set: np.ndarray):
        super().__init__(env)
        assert action_set.ndim == 2
        self.action_set = action_set.astype(np.float32)
        self.action_space = gym.spaces.Discrete(self.action_set.shape[0])

    def action(self, act_idx: int):
        return self.action_set[act_idx]


def build_ant_action_set(scale: float = 1.0) -> np.ndarray:
    a0 = np.zeros(8, dtype=np.float32)
    actions = [a0]
    for j in range(8):
        v_pos = np.zeros(8, dtype=np.float32); v_pos[j] = +scale
        v_neg = np.zeros(8, dtype=np.float32); v_neg[j] = -scale
        actions.append(v_pos); actions.append(v_neg)
    return np.stack(actions, axis=0)  # (17, 8)


def run_training():


    global_step = 0

    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    env_name = "Ant-v5"

    # Build base env once
    base = gym.make(env_name, render_mode="rgb_array", **ant_kwargs)
    

    # Build an observation_space with float32 dtype
    base_obs_space = base.observation_space
    assert isinstance(base_obs_space, Box), "Ant-v5 observation space must be Box"
    obs_space_f32 = Box(
        low=-np.inf,
        high=np.inf,
        shape=base_obs_space.shape,
        dtype=np.float32,
    )

    # Transform observation to float32 and clip actions
    base = TransformObservation(base, lambda o: np.asarray(o, dtype=np.float32), observation_space=obs_space_f32)
    base = ClipAction(base)

    # Discretize continuous actions
    action_set = build_ant_action_set(scale=1.0)
    env = DiscreteActions(base, action_set=action_set)
    env = TimeLimit(env, max_episode_steps=300)

    # Optional video
    video_folder = None
    if tm_config.record_video:
        video_folder = f"{env_name}-training"
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix="eval",
            episode_trigger=lambda ep: ep % 20 == 0,
        )

    # True sizes
    observation, _ = env.reset()
    obs_dim = int(np.array(observation, dtype=np.float32).size)
    n_actions = action_set.shape[0]  # 17

    # Agent
    dqn_agent = IQN(obs_dim=obs_dim, n_actions=n_actions, hidden_dim=256)

    # Device info
    print("=" * 50)
    print(f"Training on device: {dqn_agent.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 50)

    wandb.login(key=WANDB_API_KEY)
    run_name = f"IQN_ntau{dqn_agent.n_tau_train}-{dqn_agent.n_tau_action}_noisy"

    with wandb.init(entity = "cogitod", project="Trackmania", name=run_name, config=dqn_agent.config) as run:
        run.watch(dqn_agent.policy_network, log="all", log_freq=100)
        run.watch(dqn_agent.target_network, log="all", log_freq=100)

        tot_reward = 0.0
        tot_q_value = 0.0
        n_q_values = 0
        episode = 0

        for i in range(tm_config.training_steps):
            obs_tensor = torch.tensor(observation, dtype=torch.float32)

            eps = max(eps_end, eps_start - global_step / eps_decay)
            if np.random.rand() < eps:
                action = np.random.randint(n_actions)
                q_value = None
            else:
                 action, q_value = dqn_agent.get_action(obs_tensor.unsqueeze(0))
            if q_value is not None:
                tot_q_value += q_value
                n_q_values += 1
            
            
            global_step += 1
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done:
                print(eps)
                print(f"Episode {episode+1} total reward: {tot_reward:.2f}", flush=True)

            experience = TensorDict(
                {
                    "observation": obs_tensor,
                    "action": torch.tensor(action, dtype=torch.long),
                    "reward": torch.tensor(reward, dtype=torch.float32),
                    "next_observation": torch.tensor(next_obs, dtype=torch.float32),
                    "done": torch.tensor(done),
                },
                batch_size=torch.Size([]),
            )

            dqn_agent.store_transition(experience)
            tot_reward += float(reward)

            loss = dqn_agent.train()

            if done:
                avg_q_value = (tot_q_value / n_q_values) if n_q_values > 0 else -1.0

                log_metrics = {
                    "episode_reward": tot_reward,
                    "loss": loss,
                    "learning_rate": dqn_agent.optimizer.param_groups[0]["lr"],
                    "q_values": avg_q_value,
                }

                if tm_config.record_video and video_folder:
                    pattern = os.path.join(video_folder, "*.mp4")
                    deadline = time.time() + 2
                    video_path = None
                    while time.time() < deadline:
                        candidates = glob.glob(pattern)
                        if candidates:
                            video_path = max(candidates, key=os.path.getctime)
                            break
                    if video_path:
                        log_metrics["episode_video"] = wandb.Video(
                            video_path, format="mp4", caption=f"Episode {episode}"
                        )

                run.log(log_metrics, step=episode)

                episode += 1
                tot_reward = 0.0
                tot_q_value = 0.0
                n_q_values = 0
                observation, _ = env.reset()
            else:
                observation = next_obs

            if i % tm_config.target_network_update_frequency == 0:
                dqn_agent.update_target_network()

    env.close()


