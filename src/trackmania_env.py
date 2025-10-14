"""
Simplified Trackmania RL environment wrapper.
Single-process version based on linesight infrastructure.
"""

import cv2
import numpy as np
import torch
from pathlib import Path

from src.trackmania.tminterface import TMInterface, MessageType
from src.trackmania.map_loader import load_zone_centers, precalculate_zone_info
from config_files import tm_config


class TrackmaniaEnv:
    """
    Simplified Trackmania environment for single-process training.
    """

    def __init__(self, port: int, map_path: str, zone_centers_file: str, maps_dir: Path):
        """
        Args:
            port: TMInterface port number
            map_path: Path to the .Challenge.Gbx file
            zone_centers_file: Name of the .npy file with zone centers
            maps_dir: Directory containing map files
        """
        self.port = port
        self.map_path = map_path
        self.iface = None

        # Load map data
        self.zone_centers = load_zone_centers(
            zone_centers_file,
            maps_dir,
            tm_config.n_zone_centers_extrapolate_before_start_of_map,
            tm_config.n_zone_centers_extrapolate_after_end_of_map
        )

        self.zone_transitions, self.distances, self.cumulative_dist, self.direction_vectors = (
            precalculate_zone_info(self.zone_centers)
        )

        self.current_zone_idx = tm_config.n_zone_centers_extrapolate_before_start_of_map
        self.start_state = None
        self.episode_reward = 0
        self.episode_steps = 0

    def connect(self):
        """Connect to TMInterface."""
        if self.iface is None or not self.iface.registered:
            self.iface = TMInterface(self.port, tm_config.is_linux)
            self.iface.register(tm_config.tmi_protection_timeout_s)

            # Configure TMInterface
            self.iface.set_on_step_period(tm_config.tm_engine_step_per_action * 10)
            self.iface.set_timeout(tm_config.timeout_during_run_ms)
            self.iface.execute_command(f"cam {tm_config.game_camera_number}")

            # Load map
            self.iface.execute_command(f"map {self.map_path}")

    def reset(self):
        """Reset the environment to start a new episode."""
        self.connect()

        # Give up current race and start fresh
        self.iface.give_up()

        # Wait for race to be ready
        while True:
            msgtype = self.iface._read_int32()
            if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                race_time = self.iface._read_int32()
                if race_time == 0 and self.start_state is None:
                    self.start_state = self.iface.get_simulation_state()
                    self.iface.rewind_to_state(self.start_state)
                    self.iface._respond_to_call(msgtype)
                    break
                self.iface._respond_to_call(msgtype)
            else:
                self.iface._respond_to_call(msgtype)

        self.current_zone_idx = tm_config.n_zone_centers_extrapolate_before_start_of_map
        self.episode_reward = 0
        self.episode_steps = 0

        # Get initial observation
        obs = self._get_observation()
        return obs

    def step(self, action: int):
        """
        Execute one action in the environment.

        Args:
            action: Action index (0-3 for the default action space)

        Returns:
            observation: Current state observation
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information dict
        """
        # Set inputs based on action
        action_dict = tm_config.inputs[action]
        self.iface.set_input_state(**action_dict)

        # Step simulation forward
        self.iface.set_speed(tm_config.running_speed)

        # Wait for next step
        done = False
        reward = 0

        while True:
            msgtype = self.iface._read_int32()

            if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                race_time = self.iface._read_int32()
                self.iface._respond_to_call(msgtype)

                if race_time % (tm_config.tm_engine_step_per_action * 10) == 0:
                    # Time to compute next action
                    break

            elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                current_cp = self.iface._read_int32()
                target_cp = self.iface._read_int32()
                self.iface._respond_to_call(msgtype)

                if current_cp == target_cp:
                    # Race finished!
                    done = True
                    reward = 100  # Bonus for finishing
                    break

            else:
                self.iface._respond_to_call(msgtype)

        # Get observation
        obs = self._get_observation()

        # Calculate reward (simple distance-based reward)
        sim_state = self.iface.get_simulation_state()
        position = np.array(sim_state.dyna.current_state.position, dtype=np.float32)

        # Update zone index based on position
        for i in range(len(self.zone_centers)):
            dist = np.linalg.norm(self.zone_centers[i] - position)
            if dist < 10:  # Within 10 meters
                if i > self.current_zone_idx:
                    reward += (i - self.current_zone_idx) * 0.1  # Progress reward
                    self.current_zone_idx = i

        self.episode_reward += reward
        self.episode_steps += 1

        # Timeout after too many steps
        if self.episode_steps > 1000:
            done = True

        info = {
            "episode_reward": self.episode_reward if done else None,
            "episode_steps": self.episode_steps,
            "current_zone": self.current_zone_idx,
        }

        return obs, reward, done, info

    def _get_observation(self):
        """Get current observation from the game."""
        # Request frame from game
        self.iface.request_frame(tm_config.W_downsized, tm_config.H_downsized)

        # Wait for frame
        while True:
            msgtype = self.iface._read_int32()
            if msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                frame = self.iface.get_frame(tm_config.W_downsized, tm_config.H_downsized)
                self.iface._respond_to_call(msgtype)
                break
            else:
                self.iface._respond_to_call(msgtype)

        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        # Get simulation state for additional features
        sim_state = self.iface.get_simulation_state()
        velocity = np.array(sim_state.dyna.current_state.linear_speed, dtype=np.float32)
        speed = np.linalg.norm(velocity)

        # Package observation
        obs = {
            "image": frame_gray,  # Shape: (H, W)
            "speed": speed,  # Scalar
            "zone_idx": self.current_zone_idx,  # Scalar
        }

        return obs

    def close(self):
        """Close the environment."""
        if self.iface is not None:
            self.iface.close()
