"""
Wrappers and helper functions for Ant-v5 MuJoCo environment
"""
import numpy as np
import gymnasium as gym


class DiscreteActions(gym.ActionWrapper):
    """Wrapper to convert continuous action space to discrete action space"""
    def __init__(self, env: gym.Env, action_set: np.ndarray):
        super().__init__(env)
        assert action_set.ndim == 2
        self.action_set = action_set.astype(np.float32)
        self.action_space = gym.spaces.Discrete(self.action_set.shape[0])

    def action(self, act_idx: int):
        return self.action_set[act_idx]


def build_ant_action_set(scale: float = 1.0) -> np.ndarray:
    """
    Build discrete action set for Ant-v5 environment

    Args:
        scale: Scaling factor for actions

    Returns:
        Array of shape (17, 8) with discrete actions:
        - 1 zero action (do nothing)
        - 16 actions (8 joints × 2 directions)
    """
    a0 = np.zeros(8, dtype=np.float32)
    actions = [a0]
    for j in range(8):
        v_pos = np.zeros(8, dtype=np.float32)
        v_pos[j] = +scale
        v_neg = np.zeros(8, dtype=np.float32)
        v_neg[j] = -scale
        actions.append(v_pos)
        actions.append(v_neg)
    return np.stack(actions, axis=0)  # (17, 8)
