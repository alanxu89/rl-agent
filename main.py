import os
import datetime

from env import SimulationEnv
from trainer import Trainer
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage


class Config:

    def __init__(self):
        self.seed = 0

        self.num_workers = 1
        self.discount = 0.997

        # Training
        #  Path to store the model weights and TensorBoard logs
        print(os.path.dirname(os.path.realpath(__file__)))
        self.results_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "/results",
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        print(self.results_path)
        self.save_model = True
        self.training_steps = int(1e6)
        self.batch_size = 256
        # Number of training steps before using the model for self-playing
        self.checkpoint_interval = int(1e3)
        # Scale the value loss to avoid overfitting of the value function,
        # paper recommends 0.25 (See paper appendix Reanalyze)
        self.value_loss_weight = 0.25
        self.train_on_gpu = True

        self.optimizer = "Adam"

        self.lr_init = 0.001
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 10000

        # Replay Buffer
        self.replay_buffer_size = int(1e6)
        # Number of steps in the future to take into account for calculating the target value
        self.td_steps = 10


if __name__ == "__main__":
    print(os.path.realpath(__file__))
    config = Config()
