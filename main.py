import os
import datetime
import time
import pickle
import copy

import tensorflow as tf
import numpy as np
import ray

from env import SimulationEnv
from trainer import Trainer, CPUActor
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage
from sim_player import SimPlayer


class Config:

    def __init__(self):
        self.seed = 0

        self.num_workers = 2
        self.discount = 0.997

        # Training
        #  Path to store the model weights and TensorBoard logs
        print(os.path.dirname(os.path.realpath(__file__)))
        self.results_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "results",
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        print("print results path", self.results_path)
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

        self.play_delay = False

        self.ratio = False


class RLAgent:

    def __init__(self, config=None):
        self.config = Config()

        if config:
            if type(config) is dict:
                for key, value in config.items():
                    setattr(self.config, key, value)
            else:
                self.config = config

        self.num_gpus = len(tf.config.list_physical_devices('GPU'))
        ray.init(num_gpus=self.num_gpus, ignore_reinit_error=True)

        np.random.seed(self.config.seed)
        tf.random.set_seed(self.config.seed)

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "state_encoder_weights": None,
            "actor_weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "terminate": False,
        }

        self.replay_buffer = {}

        cpu_actor = CPUActor.remote()
        cpu_weights = ray.get(cpu_actor.get_initial_weights.remote(
            self.config))
        self.checkpoint["state_encoder_weights"] = cpu_weights[0]
        self.checkpoint["actor_weights"] = cpu_weights[1]

        # Workers
        self.sim_workers = None
        self.test_worker = None
        self.training_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def train(self, log_in_tensorboard=True):
        if log_in_tensorboard or self.config.save_model:
            os.makedirs(self.config.results_path, exist_ok=True)

        self.shared_storage_worker = SharedStorage.remote(
            self.config, self.checkpoint)
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = ReplayBuffer.remote(
            self.config.replay_buffer_size)

        # self.training_worker = Trainer.options(num_cpus=0, num_gpus=1).remote(
        #     self.config, self.replay_buffer_worker, self.shared_storage_worker)

        self.sim_workers = [
            SimPlayer.options(num_cpus=1,
                              num_gpus=1).remote(self.config,
                                                 self.config.seed + seed)
            for seed in range(self.config.num_workers)
        ]

        # launch self play
        for sim_worker in self.sim_workers:
            sim_worker.continuous_play.remote(self.shared_storage_worker,
                                              self.replay_buffer_worker)

        # self.training_worker.train.remote()

        for i in range(100):
            time.sleep(1)
        # if log_in_tensorboard:
        #     self.logging_loop()

    def logging_loop(self):

        hyper_params_table = [
            f"| {key} | {value} |"
            for key, value in self.config.__dict__.items()
        ]
        counter = 0

        with tf.device('/CPU'):
            writer = tf.summary.create_file_writer("/tmp/mylogs/eager")
            with writer.as_default():
                tf.summary.text(
                    "Hyperparameters",
                    "| Parameter | Value |\n|-------|-------|\n" +
                    "\n".join(hyper_params_table), counter)

                tf.summary.text("model summary", "", counter)

        keys = [
            "total_reward",
            "muzero_reward",
            "opponent_reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "num_played_games",
            "num_played_steps",
            "num_reanalysed_games",
        ]
        info = self.shared_storage_worker.get_info.remote(keys)
        try:
            while info["training_step"] < self.config.training_steps:
                info = self.shared_storage_worker.get_info.remote(keys)
                # details to be implemented
                with tf.device('/CPU'):
                    with writer.as_default():
                        tf.summary.scalar("3.Loss/Reward_loss",
                                          info["reward_loss"], counter)
                counter += 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

        if self.config.save_model:
            pickle.dump(
                {
                    "buffer": self.replay_buffer,
                    "num_played_games": self.checkpoint["num_played_games"],
                    "num_played_steps": self.checkpoint["num_played_steps"],
                },
                open(
                    os.path.join(self.config.results_path,
                                 "replay_buffer.pkl"), "wb"),
            )

    def terminate_workers(self):
        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def test(self, render=True, num_tests=1):
        sim_worker = SimPlayer(self.config, np.random.randint(10000))
        results = []

        for i in range(num_tests):
            print(f"Testing {i+1}/{num_tests}")
            results.append(sim_worker.play(render))
        sim_worker.close_game()

        if len(self.config.players) == 1:
            result = np.mean(
                [sum([rd.reward for rd in res]) for res in results])

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                self.checkpoint = tf.keras.models.load_model(checkpoint_path)
                print(f"\nUsing checkpoint from {checkpoint_path}")
            else:
                print(f"\nThere is no model saved in {checkpoint_path}.")

        if replay_buffer_path:
            if os.path.exists(replay_buffer_path):
                with open(replay_buffer_path, "rb") as f:
                    replay_buffer_infos = pickle.load(f)
                self.replay_buffer = replay_buffer_infos["replay_buffer"]
                self.checkpoint["num_played_steps"] = replay_buffer_infos[
                    "num_played_steps"]
                self.checkpoint["num_played_games"] = replay_buffer_infos[
                    "num_played_games"]
                self.checkpoint["num_reanalysed_games"] = replay_buffer_infos[
                    "num_reanalysed_games"]

                print(
                    f"\nInitializing replay buffer with {replay_buffer_path}")
            else:
                print(
                    f"Warning: Replay buffer path '{replay_buffer_path}' doesn't exist.  Using empty buffer."
                )
                self.checkpoint["training_step"] = 0
                self.checkpoint["num_played_steps"] = 0
                self.checkpoint["num_played_games"] = 0
                self.checkpoint["num_reanalysed_games"] = 0


if __name__ == "__main__":
    print(os.path.realpath(__file__))
    config = Config()
    rlagent = RLAgent(config)
    rlagent.train()
