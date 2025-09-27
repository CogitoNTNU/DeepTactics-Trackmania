from sys import platform

is_linux = platform in ["linux", "linux2"]

training_steps = 1_000_000
target_network_update_frequency = 500