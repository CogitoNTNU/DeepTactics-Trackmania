
import numpy as np

#kanskje fjerne? flytte direkte til config
number_of_actions = 13

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

#####################################
#oppdater number of actions i toppen