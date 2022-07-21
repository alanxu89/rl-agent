import time
import copy

import ray
import numpy as np
import tensorflow as tf

from models import RepresentationNetwork, ActorNetwork, CriticNetwork
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

    def __init__(self, config, replay_buffer: ReplayBuffer,
                 shared_storage: SharedStorage):
        self.config = config

        np.random.seed(self.config.seed)
        tf.random.set_seed(self.config.seed)

        self.state_encoder = RepresentationNetwork()

        self.critic = CriticNetwork()
        self.actor = ActorNetwork()

        self.critic_target = CriticNetwork()
        self.actor_target = ActorNetwork()

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.config.lr_init,
            self.config.lr_decay_steps,
            self.config.lr_decay_rate,
            staircase=True)

        self.optimizer = tf.keras.optimizers.Adam(self.lr_schedule)

        self.tau = 0.1
        self.step = 0

        self.replay_buffer = replay_buffer
        self.shared_storage = shared_storage

    def train(self):

        # Wait for the replay buffer to be filled
        while ray.get(
                self.shared_storage.get_info.remote(
                    "num_played_episodes")) < 10:
            print("sleep 1")
            time.sleep(1)

        while self.step < self.config.training_steps and not ray.get(
                self.shared_storage.get_info.remote("terminate")):
            print("training steps {} and terminate status {}".format(
                self.step,
                ray.get(self.shared_storage.get_info.remote("terminate"))))

            batch = ray.get(self.replay_buffer.get_batch.remote())
            self.train_step(batch)

            if self.step % self.config.checkpoint_interval == 0:
                self.shared_storage.set_info({
                    "state_encoder_weights":
                    copy.deepcopy(self.state_encoder.get_weights()),
                    "actor_weights":
                    copy.deepcopy(self.actor.get_weights()),
                    "optimizer_state":
                    tf.keras.optimizers.serialize(self.optimizer)
                })
                if self.config.save_model:
                    self.shared_storage.save_checkpoint()

    def train_step(self, replay_data):
        """ TD3 train step
        https://spinningup.openai.com/en/latest/algorithms/td3.html#pseudocode
        https://zhuanlan.zhihu.com/p/357719456
        """
        (observations, actions, next_observations, rewards,
         dones) = replay_data

        actor_losses, critic_losses = [], []

        self.step += 1
        # Sample replay buffer

        # Select action according to policy and add clipped noise
        noise = tf.clip_by_value(tf.random.normal(tf.shape(actions)), -0.1,
                                 0.1)
        next_actions = tf.clip_by_value(
            self.actor_target(self.state_encoder(next_observations)) + noise,
            -1, 1)

        # Compute the next Q-values: min over all critics targets
        q1_target, q2_target = self.critic_target(
            self.state_encoder(next_observations), next_actions)
        next_q_values = tf.minimum(q1_target, q2_target)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        with tf.GradientTape() as critic_tape:
            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(self.state_encoder(observations),
                                           actions)
            # Compute critic loss
            critic_loss = sum([
                tf.reduce_mean(tf.square(current_q - target_q_values))
                for current_q in current_q_values
            ])
            critic_losses.append(critic_loss.item())

        grads = critic_tape.gradient(
            critic_loss, self.critic.trainable_weights +
            self.state_encoder.trainable_weights)
        self.optimizer.apply_gradients(
            zip(
                grads, self.critic.trainable_weights +
                self.state_encoder.trainable_weights))

        # Delayed policy updates
        if self.step % self.policy_delay == 0:
            with tf.GradientTape() as actor_tape:
                # Compute actor loss
                encoded_state = self.state_encoder(observations)
                q1, q2 = self.critic(encoded_state, self.actor(encoded_state))
                actor_loss = -q1
                actor_losses.append(actor_loss)

            grads = actor_tape.gradient(actor_loss,
                                        self.actor.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads, self.actor.trainable_weights))

            for param, target_param in zip(
                    self.critic.trainable_weights,
                    self.critic_target.trainable_weights):
                target_param.assign(self.tau * param +
                                    (1 - self.tau) * target_param)

            for param, target_param in zip(
                    self.actor.trainable_weights,
                    self.actor_target.trainable_weights):
                target_param.assign(self.tau * param +
                                    (1 - self.tau) * target_param)


@ray.remote(num_cpus=1, num_gpus=0)
class CPUActor:

    def __init__(self):
        pass

    def get_initial_weights(self, config):
        with tf.device('/CPU:0'):
            state_encoder = RepresentationNetwork()
            actor = ActorNetwork()

        return state_encoder.get_weights(), actor.get_weights()


if __name__ == "__main__":
    trainer = Trainer()
