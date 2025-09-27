from dataclasses import dataclass
import torch
from typing import Optional

@dataclass
class Experience:
    state: torch.Tensor
    next_state: Optional[torch.Tensor]
    action: int
    done: bool
    reward: float