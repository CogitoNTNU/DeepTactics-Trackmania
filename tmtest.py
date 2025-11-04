from src.trackmania_interface.game_instance_manager import GameInstanceManager
from torch.multiprocessing import Lock
import time

from src.trackmania_interface.tminterface2 import TMInterface


print("Started")
test_game = GameInstanceManager(game_spawning_lock=Lock(), tmi_port=8478)
print("Initiated")

test_game.iface = TMInterface(8478)

test_game.launch_game()

test_game.iface.register()  #

test_game.request_map("ESL-Hockolicious.Challenge.Gbx", None)
print("Lmao")
time.sleep(7)
print("Lmao2")
for _ in range(20):
    print(test_game.grab_screen())
    test_game.iface.set_input_state(False, False, True, False)

