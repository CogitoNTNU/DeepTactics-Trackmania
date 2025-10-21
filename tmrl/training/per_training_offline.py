"""
TrainingOffline extension that enables PER memory integration.

This ensures the training agent gets a reference to the memory for priority updates.
"""

from tmrl.training_offline import TrainingOffline


class PERTrainingOffline(TrainingOffline):
    """
    Extended TrainingOffline that connects PER memory to training agent.

    This class ensures the training agent can update priorities in the replay buffer.
    """

    def __init__(self, *args, **kwargs):
        """Initialize and connect memory to training agent."""
        super().__init__(*args, **kwargs)

        # Set memory reference in agent if it has set_memory method
        if hasattr(self.agent, 'set_memory'):
            self.agent.set_memory(self.memory)
            print("[PER] Connected prioritized memory to training agent")
        else:
            print("[Warning] Training agent does not support set_memory()")
