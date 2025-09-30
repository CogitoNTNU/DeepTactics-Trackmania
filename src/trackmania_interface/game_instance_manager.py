"""
This file implements the main logic to interact with the game, via the GameInstanceManager class.

The entry point is the rollout() function.

Given a map and a policy, this function will:
    - if necessary, start or restart the game
    - if necessary, load the map
    - start a Trackmania run
    - apply the policy at every step
    - collect various statistics
    - stop the run if the race is finished or the agent is not making progress anymore
    - return the results of the full run which can be used to train the agent elsewhere

/!\
This file has first been implemented for TMI 1.4.3 which was neither suited for regular interruptions forced by the
exploration policy nor to synchronize visual frames and game engine steps. Through small incremental modifications, the
file has become complex.
It is possible that some parts of the file are not necessary since the transition to TMI 2.1.4. We have chosen to focus
our attention elsewhere, because "If it works don't touch it.".
Contributions are welcome to simplify this part of the code.
"""

import math
import os
import socket
import subprocess
import time
from typing import Callable, Dict, List

import cv2
import numba
import numpy as np
import numpy.typing as npt
import psutil

from config_files import tm_config, user_config
from src import contact_materials, map_loader
from src.trackmania_interface.tminterface2 import MessageType, TMInterface

if tm_config.is_linux:
    from xdo import Xdo
else:
    import win32.lib.win32con as win32con
    import win32com.client
    import win32gui
    import win32process


def _set_window_focus(trackmania_window):
    # https://stackoverflow.com/questions/14295337/win32gui-setactivewindow-error-the-specified-procedure-could-not-be-found
    if tm_config.is_linux:
        Xdo().activate_window(trackmania_window)
    else:
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys("%")
        win32gui.SetForegroundWindow(trackmania_window)


def ensure_not_minimized(trackmania_window):
    if tm_config.is_linux:
        Xdo().map_window(trackmania_window)
    else:
        if win32gui.IsIconic(
            trackmania_window
        ):  # https://stackoverflow.com/questions/54560987/restore-window-without-setting-to-foreground
            win32gui.ShowWindow(trackmania_window, win32con.SW_SHOWNORMAL)  # Unminimize window


@numba.njit
def update_current_zone_idx(
    current_zone_idx: int,
    zone_centers: npt.NDArray,
    sim_state_position: npt.NDArray,
    max_allowable_distance_to_virtual_checkpoint: float,
    next_real_checkpoint_positions: npt.NDArray,
    max_allowable_distance_to_real_checkpoint: npt.NDArray,
):
    d1 = np.linalg.norm(zone_centers[current_zone_idx + 1] - sim_state_position)
    d2 = np.linalg.norm(zone_centers[current_zone_idx] - sim_state_position)
    d3 = np.linalg.norm(zone_centers[current_zone_idx - 1] - sim_state_position)
    d4 = np.linalg.norm(next_real_checkpoint_positions[current_zone_idx] - sim_state_position)
    while (
        d1 <= d2
        and d1 <= max_allowable_distance_to_virtual_checkpoint
        and current_zone_idx
        < len(zone_centers) - 1 - tm_config.n_zone_centers_extrapolate_after_end_of_map  # We can never enter the final virtual zone
        and d4 < max_allowable_distance_to_real_checkpoint[current_zone_idx]
    ):
        # Move from one virtual zone to another
        current_zone_idx += 1
        d2, d3 = d1, d2
        d1 = np.linalg.norm(zone_centers[current_zone_idx + 1] - sim_state_position)
        d4 = np.linalg.norm(next_real_checkpoint_positions[current_zone_idx] - sim_state_position)
    while current_zone_idx >= 2 and d3 < d2 and d3 <= max_allowable_distance_to_virtual_checkpoint:
        current_zone_idx -= 1
        d1, d2 = d2, d3
        d3 = np.linalg.norm(zone_centers[current_zone_idx - 1] - sim_state_position)
    return current_zone_idx

#MARK:GIM
class GameInstanceManager:
    def __init__(
        self,
        game_spawning_lock,
        running_speed=1,
        run_steps_per_action=10,
        max_overall_duration_ms=2000,
        max_minirace_duration_ms=2000,
        tmi_port=None,
    ):
        # Create TMInterface we will be using to interact with the game client
        self.iface = None
        self.latest_tm_engine_speed_requested = 1
        self.running_speed = running_speed
        self.run_steps_per_action = run_steps_per_action
        self.max_overall_duration_ms = max_overall_duration_ms
        self.max_minirace_duration_ms = max_minirace_duration_ms
        self.timeout_has_been_set = False
        self.msgtype_response_to_wakeup_TMI = None
        self.latest_map_path_requested = -2
        self.last_rollout_crashed = False
        self.last_game_reboot = time.perf_counter()
        self.UI_disabled = False
        self.tmi_port = tmi_port
        self.tm_process_id = None
        self.tm_window_id = None
        self.start_states = {}
        self.game_spawning_lock = game_spawning_lock
        self.game_activated = False

    def get_tm_window_id(self):
        assert self.tm_process_id is not None

        if tm_config.is_linux:
            self.tm_window_id = None
            while self.tm_window_id is None:  # This outer while is for the edge case where the window may not have had time to be launched
                window_search_depth = 1
                while True:  # This inner while is to try and find the right depth of the window in Xdo().search_windows()
                    c1 = set(Xdo().search_windows(winname=b"TrackMania Modded", max_depth=window_search_depth + 1))
                    c2 = set(Xdo().search_windows(winname=b"TrackMania Modded", max_depth=window_search_depth))
                    c1 = {w_id for w_id in c1 if Xdo().get_pid_window(w_id) == self.tm_process_id}
                    c2 = {w_id for w_id in c2 if Xdo().get_pid_window(w_id) == self.tm_process_id}
                    c1_diff_c2 = c1.difference(c2)
                    if len(c1_diff_c2) == 1:
                        self.tm_window_id = c1_diff_c2.pop()
                        break
                    elif (
                        len(c1_diff_c2) == 0 and len(c1) > 0
                    ) or window_search_depth >= 10:  # 10 is an arbitrary cutoff in this search we do not fully understand
                        print(
                            "Warning: Worker could not find the window of the game it just launched, stopped at window_search_depth",
                            window_search_depth,
                        )
                        break
                    window_search_depth += 1
        else:

            def get_hwnds_for_pid(pid):
                def callback(hwnd, hwnds):
                    _, found_pid = win32process.GetWindowThreadProcessId(hwnd)

                    if found_pid == pid:
                        hwnds.append(hwnd)
                    return True

                hwnds = []
                win32gui.EnumWindows(callback, hwnds)
                return hwnds

            while True:
                for hwnd in get_hwnds_for_pid(self.tm_process_id):
                    if win32gui.GetWindowText(hwnd).startswith("Track"):
                        self.tm_window_id = hwnd
                        return
                # else:
                #     raise Exception("Could not find TmForever window id.")

    def is_tm_process(self, process: psutil.Process) -> bool:
        try:
           print(process.name())
           return process.name().startswith("TmForever")
        #try: 
        #    return process.name().startswith("TrackMania")
        except psutil.NoSuchProcess:
            return False

    def get_tm_pids(self) -> List[int]:
        return [process.pid for process in psutil.process_iter() if self.is_tm_process(process)]

#MARK: Luanch
    def launch_game(self):
        self.tm_process_id = None

        if tm_config.is_linux:
            self.game_spawning_lock.acquire()
            pid_before = self.get_tm_pids()
            os.system(str(user_config.linux_launch_game_path) + " " + str(self.tmi_port))
            while True:
                pid_after = self.get_tm_pids()
                tmi_pid_candidates = set(pid_after) - set(pid_before)
                if len(tmi_pid_candidates) > 0:
                    assert len(tmi_pid_candidates) == 1
                    break
            self.tm_process_id = list(tmi_pid_candidates)[0]
        else:
            launch_string = (
                'powershell -executionPolicy bypass -command "& {'
                f" $process = start-process -FilePath '{user_config.windows_TMLoader_path}'"
                " -PassThru -ArgumentList "
                f'\'run TmForever "{user_config.windows_TMLoader_profile_name}" /configstring=\\"set custom_port {self.tmi_port}\\"\';'
                ' echo exit $process.id}"'
            )

            tmi_process_id = int(subprocess.check_output(launch_string).decode().split("\r\n")[1])
            while self.tm_process_id is None:
                tm_processes = list(
                    filter(
                        lambda s: s.startswith("TmForever"),
                        subprocess.check_output("wmic process get Caption,ParentProcessId,ProcessId").decode().split("\r\n"),
                    )
                )
                for process in tm_processes:
                    name, parent_id, process_id = process.split()
                    parent_id = int(parent_id)
                    process_id = int(process_id)
                    if parent_id == tmi_process_id:
                        self.tm_process_id = process_id
                        break

        print(f"Found Trackmania process id: {self.tm_process_id=}")
        self.last_game_reboot = time.perf_counter()
        self.latest_map_path_requested = -1
        self.msgtype_response_to_wakeup_TMI = None
        while not self.is_game_running():
            time.sleep(0)

        self.get_tm_window_id()

    def is_game_running(self):
        return (self.tm_process_id is not None) and (self.tm_process_id in (p.pid for p in psutil.process_iter()))

    def close_game(self):
        self.timeout_has_been_set = False
        self.game_activated = False
        assert self.tm_process_id is not None
        if tm_config.is_linux:
            os.system("kill -9 " + str(self.tm_process_id))
        else:
            os.system(f"taskkill /PID {self.tm_process_id} /f")
        while self.is_game_running():
            time.sleep(0)

    def ensure_game_launched(self):
        if not self.is_game_running():
            print("Game not found. Restarting TMInterface.")
            self.launch_game()

    def grab_screen(self):
        return self.iface.get_frame(tm_config.W_downsized, tm_config.H_downsized)

    def request_speed(self, requested_speed: float):
        self.iface.set_speed(requested_speed)
        self.latest_tm_engine_speed_requested = requested_speed

    def request_inputs(self, action_idx: int, rollout_results: Dict):
        self.iface.set_input_state(**tm_config.inputs[action_idx])

    def request_map(self, map_path: str, zone_centers: npt.NDArray):
        self.latest_map_path_requested = map_path
        if user_config.is_linux:
            map_path = map_path.replace("\\", "/")
        else:
            map_path = map_path.replace("/", "\\")
        map_loader.hide_personal_record_replay(map_path, True)
        self.iface.execute_command(f"map {map_path}")
        self.UI_disabled = False
        #(
            #self.next_real_checkpoint_positions,
            #self.max_allowable_distance_to_real_checkpoint,
        #) = map_loader.sync_virtual_and_real_checkpoints(zone_centers, map_path)

    #MARK: Run race