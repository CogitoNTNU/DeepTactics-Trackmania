# MessageType Documentation

## Overview

`MessageType` is an `IntEnum` that defines message types for socket-based communication between Python and TMInterface 2.1.4. It enables bidirectional communication for controlling TrackMania simulations.

Located in: `trackmania_rl/tmi_interaction/tminterface2.py:21`

## Message Categories

### Server-to-Client Messages (SC_*)

Messages sent FROM TMInterface TO Python:

| Message Type | Purpose | When Triggered |
|-------------|---------|----------------|
| `SC_RUN_STEP_SYNC` | Simulation step update | Every simulation tick |
| `SC_CHECKPOINT_COUNT_CHANGED_SYNC` | Checkpoint event | When car passes a checkpoint |
| `SC_LAP_COUNT_CHANGED_SYNC` | Lap completion | When a lap is completed |
| `SC_REQUESTED_FRAME_SYNC` | Frame data ready | When frame capture is requested |
| `SC_ON_CONNECT_SYNC` | Connection established | Initial connection to TMInterface |

### Client-to-Server Messages (C_*)

Commands sent FROM Python TO TMInterface:

| Message Type | Purpose | Parameters |
|-------------|---------|------------|
| `C_SET_SPEED` | Set simulation speed | float: speed multiplier |
| `C_REWIND_TO_STATE` | Restore simulation state | int: state_length, bytes: state_data |
| `C_REWIND_TO_CURRENT_STATE` | Rewind to saved state | None |
| `C_GET_SIMULATION_STATE` | Request current state | None |
| `C_SET_INPUT_STATE` | Set car controls | 4x uint8: left, right, accel, brake |
| `C_GIVE_UP` | Abandon current run | None |
| `C_PREVENT_SIMULATION_FINISH` | Keep simulation running | None |
| `C_SHUTDOWN` | Close connection | None |
| `C_EXECUTE_COMMAND` | Run TMInterface command | int: length, string: command |
| `C_SET_TIMEOUT` | Set response timeout | uint32: timeout_ms |
| `C_RACE_FINISHED` | Check if race ended | None |
| `C_REQUEST_FRAME` | Request frame capture | 2x int32: width, height |
| `C_UNREQUEST_FRAME` | Stop frame capture | None |
| `C_RESET_CAMERA` | Reset camera position | None |
| `C_SET_ON_STEP_PERIOD` | Set step callback interval | int32: period |
| `C_TOGGLE_INTERFACE` | Show/hide TMInterface UI | int32: bool (0 or 1) |
| `C_IS_IN_MENUS` | Check if in menu | None |
| `C_GET_INPUTS` | Get current input string | None |

## Usage Examples

### Sending Commands (Client → Server)

```python
import struct
import numpy as np
from tminterface2 import MessageType, TMInterface

iface = TMInterface(port=8476)

# Simple command (no parameters)
iface.sock.sendall(struct.pack("i", MessageType.C_SHUTDOWN))

# Command with single parameter
iface.sock.sendall(struct.pack("if", MessageType.C_SET_SPEED, np.float32(2.0)))

# Command with multiple parameters
iface.sock.sendall(
    struct.pack("iBBBB",
                MessageType.C_SET_INPUT_STATE,
                np.uint8(left),
                np.uint8(right),
                np.uint8(accelerate),
                np.uint8(brake))
)

# Command with variable-length data
command = "set countdown_speed 10.0"
iface.sock.sendall(struct.pack("ii", MessageType.C_EXECUTE_COMMAND, np.int32(len(command))))
iface.sock.sendall(command.encode())
```

### Receiving Messages (Server → Client)

```python
# Read message type
msgtype = iface._read_int32()

# Handle different message types
if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
    time = iface._read_int32()
    # Process simulation step at this time
    iface._respond_to_call(msgtype)

elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
    current = iface._read_int32()
    target = iface._read_int32()
    # Handle checkpoint event
    iface._respond_to_call(msgtype)

elif msgtype == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):
    current_lap = iface._read_int32()
    target_lap = iface._read_int32()
    # Handle lap completion
    iface._respond_to_call(msgtype)

elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
    frame = iface.get_frame(width, height)
    # Process frame data
    iface._respond_to_call(msgtype)

elif msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
    # Initialize connection settings
    iface.set_speed(1.0)
    iface.set_on_step_period(100)
    iface._respond_to_call(msgtype)
```

## Message Protocol

### Binary Format

All messages use little-endian byte order via Python's `struct.pack()`:

- `"i"` - signed 32-bit integer
- `"I"` - unsigned 32-bit integer
- `"f"` - 32-bit float
- `"B"` - unsigned 8-bit integer (byte)

### Response Pattern

For server-to-client messages, you typically:
1. Read the message type (int32)
2. Read any associated parameters
3. Process the message
4. Call `iface._respond_to_call(msgtype)` to acknowledge

## Common Patterns

### Setting Car Inputs
```python
def set_inputs(iface, left, right, accel, brake):
    """Set car control inputs (all boolean values)."""
    iface.sock.sendall(
        struct.pack("iBBBB",
                    MessageType.C_SET_INPUT_STATE,
                    np.uint8(left),
                    np.uint8(right),
                    np.uint8(accel),
                    np.uint8(brake))
    )
```

### Main Communication Loop
```python
while running:
    msgtype = iface._read_int32()

    if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
        time = iface._read_int32()
        # Your step logic here
        iface._respond_to_call(msgtype)

    elif msgtype == int(MessageType.C_SHUTDOWN):
        break
```

## See Also

- `TMInterface` class in `trackmania_rl/tmi_interaction/tminterface2.py`
- `GameInstanceManager` usage example in `trackmania_rl/tmi_interaction/game_instance_manager.py`
- TMInterface 2.1.4 documentation
