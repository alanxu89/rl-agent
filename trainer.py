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


# @ray.remote
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
        while ray.get(shared_storage.get_info("num_played_games")) < 1:
            time.sleep(1)

        next_batch = replay_buffer.get_batch()

        while self.training_step < self.config.training_steps and not ray.get(
                shared_storage.get_info("terminate")):
            print("training steps {} and terminate status {}".format(
                self.training_step,
                ray.get(shared_storage.get_info("terminate"))))
            batch = replay_buffer.get_batch()
            priorities, total_loss, value_loss, reward_loss, policy_loss = self.update_weights(
                batch)

            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info({
                    "weights":
                    copy.deepcopy(self.model.get_weights()),
                    "optimizer_state":
                    tf.keras.optimizers.serialize(self.optimizer)
                })
                if self.config.save_model:
                    shared_storage.save_checkpoint()
            shared_storage.set_info({
                "training_step": self.training_step,
                "lr": self.optimizer.get_config()["lr"],
                "total_loss": total_loss,
                "value_loss": value_loss,
                "reward_loss": reward_loss,
                "policy_loss": policy_loss,
            })

    def train_step(self, replay_data):
        """ TD3 train step
        """
        actor_losses, critic_losses = [], []

        self.training_step += 1
        # Sample replay buffer

        # Select action according to policy and add clipped noise
        noise = replay_data.actions.clone().data.normal_(
            0, self.target_policy_noise)
        noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
        next_actions = (self.actor_target(replay_data.next_observations) +
                        noise).clamp(-1, 1)

        # Compute the next Q-values: min over all critics targets
        next_q_values = tf.concat(self.critic_target(
            replay_data.next_observations, next_actions),
                                  dim=1)
        next_q_values, _ = tf.min(next_q_values, dim=1, keepdim=True)
        target_q_values = replay_data.rewards + (
            1 - replay_data.dones) * self.gamma * next_q_values

        with tf.GradientTape() as critic_tape:
            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations,
                                           replay_data.actions)
            # Compute critic loss
            critic_loss = sum([
                tf.reduce_mean(tf.square(current_q - target_q_values))
                for current_q in current_q_values
            ])
            critic_losses.append(critic_loss.item())

        grads = critic_tape.gradient(critic_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_weights))

        # Delayed policy updates
        if self.training_step % self.policy_delay == 0:
            with tf.GradientTape() as actor_tape:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations,
                    self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

            grads = actor_tape.gradient(actor_loss,
                                        self.actor.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads, self.actor.trainable_weights))


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, lr_init, lr_decay_rate, lr_decay_steps):
        super().__init__()
        self.lr_init = lr_init
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps

    def __call__(self, step):
        return self.lr_init * self.lr_decay_rate**(step / self.lr_decay_steps)
