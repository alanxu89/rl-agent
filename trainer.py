import time
import copy

import ray
import numpy as np
import tensorflow as tf

from models import BaselinePolicyNet
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage


def set_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


@ray.remote
class Trainer:

    def __init__(self, config, initial_checkpoint):
        self.config = config

        np.random.seed(self.config.seed)
        tf.random.set_seed(self.config.seed)

        self.model = BaselinePolicyNet()
        initial_weights = copy.deepcopy(initial_checkpoint["weights"])
        if initial_weights:
            self.model.set_weights(initial_weights)

        self.training_step = initial_checkpoint["training_step"]

        self.lr_schedule = CustomSchedule(self.config.lr_init,
                                          self.config.lr_decay_rate,
                                          self.config.lr_decay_steps)

        self.optimizer = tf.keras.optimizers.Adam(
            self.lr_schedule
            # weight_decay=self.config.weight_decay,
        )

        if initial_checkpoint["optimizer_state"] is not None:
            self.optimizer.set_weights(
                copy.deepcopy(initial_checkpoint["optimizer_state"]))

    def continuous_update_weights(self, replay_buffer: ReplayBuffer,
                                  shared_storage: SharedStorage):

        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(1)

        next_batch = replay_buffer.get_batch.remote()

        while self.training_step < self.config.training_steps and not ray.get(
                shared_storage.get_info.remote("terminate")):
            print("training steps {} and terminate status {}".format(
                self.training_step,
                ray.get(shared_storage.get_info.remote("terminate"))))
            batch = replay_buffer.get_batch.remote()
            priorities, total_loss, value_loss, reward_loss, policy_loss = self.update_weights(
                batch)

            if self.config.PER:
                # to be implemented
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                # replay_buffer.update_priorities.remote(priorities, index_batch)
                a = 1

            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote({
                    "weights":
                    copy.deepcopy(self.model.get_weights()),
                    "optimizer_state":
                    tf.keras.optimizers.serialize(self.optimizer)
                })
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()
            shared_storage.set_info.remote({
                "training_step":
                self.training_step,
                "lr":
                self.optimizer.get_config()["lr"],
                "total_loss":
                total_loss,
                "value_loss":
                value_loss,
                "reward_loss":
                reward_loss,
                "policy_loss":
                policy_loss,
            })

    def train_step(self, batch):
        """ train step """
        (
            observation_batch,
            action_batch,
            target_reward,
            target_policy,
        ) = batch

        with tf.GradientTape() as tape:
            if self.config.PER:
                weight_batch = tf.Tensor(weight_batch.copy())

            actions = self.model(observation_batch)

            seq_len = action_batch.shape[-1]

            loss = 0

            if self.config.PER:
                loss *= weight_batch
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_weights))
        self.training_step += 1


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, lr_init, lr_decay_rate, lr_decay_steps):
        super().__init__()
        self.lr_init = lr_init
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps

    def __call__(self, step):
        return self.lr_init * self.lr_decay_rate**(step / self.lr_decay_steps)
