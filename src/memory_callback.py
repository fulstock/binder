from transformers import TrainerCallback
import torch
import logging

logger = logging.getLogger(__name__)

class MemoryUsageCallback(TrainerCallback):
    """
    A custom callback to monitor and log GPU memory usage during training.
    """

    def __init__(self):
        super().__init__()
        self.max_memory_allocated = 0
        self.max_memory_reserved = 0

    def _log_memory_usage(self, event: str):
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device_index)
            reserved = torch.cuda.memory_reserved(device_index)

            # Update max memory usage
            self.max_memory_allocated = max(self.max_memory_allocated, allocated)
            self.max_memory_reserved = max(self.max_memory_reserved, reserved)
            
            allocated_gb = allocated / (1024 ** 3)
            reserved_gb = reserved / (1024 ** 3)

            logger.info(
                f"📊 GPU memory usage {event} – allocated: {allocated_gb:.2f} GB | reserved: {reserved_gb:.2f} GB (device {device_index})"
            )

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        self._log_memory_usage("at end of epoch")

    def on_evaluate(self, args, state, control, **kwargs):
        """Called before and after each evaluation."""
        self._log_memory_usage("at evaluation step")

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        max_allocated_gb = self.max_memory_allocated / (1024 ** 3)
        max_reserved_gb = self.max_memory_reserved / (1024 ** 3)
        logger.info(
            f"📈 Max GPU memory usage during training – allocated: {max_allocated_gb:.2f} GB | reserved: {max_reserved_gb:.2f} GB"
        ) 