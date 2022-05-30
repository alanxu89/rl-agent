import time
import math

import ray
import numpy as np
import tensorflow as tf

from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage
from models import RepresentationNetwork, ActorNetwork
from env import SimulationEnv


@ray.remote
class LogPlayer:

    def __init__(self, config, seed):
        self.config = config
        self.data_files = [
            "/home/alanxu/Downloads/waymo_motion_data/"
            "uncompressed_scenario_training_training.tfrecord-00491-of-01000"
        ]
        self.env = SimulationEnv(self.data_files)

        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.state_enc = RepresentationNetwork()
        self.actor = ActorNetwork()

    def continuous_play(self,
                        shared_storage: SharedStorage,
                        replay_buffer: ReplayBuffer,
                        test_mode=False):
        while ray.get(shared_storage.get_info.remote(
                "training_step")) < self.config.training_steps and not ray.get(
                    shared_storage.get_info.remote("terminate")):
            self.state_enc.set_weights(
                ray.get(
                    shared_storage.get_info.remote("state_encoder_weights")))
            self.actor.set_weights(
                ray.get(shared_storage.get_info.remote("actor_weights")))

            if not test_mode:
                replay_data = self.play(False)
                for rd in replay_data:
                    replay_buffer.add.remote(rd.observation, rd.action,
                                             rd.next_observation, rd.reward,
                                             rd.done, {})

            else:
                replay_data = self.play(False)
                shared_storage.set_info.remote({
                    "episode_length":
                    len(len(replay_data)) - 1,
                    "total_reward":
                    sum([rd.reward for rd in replay_data]),
                })

            if not test_mode and self.config.play_delay:
                time.sleep(self.config.play_delay)
            if not test_mode and self.config.ratio:
                ratio = ray.get(
                    shared_storage.get_info.remote("training_step")) / max(
                        1,
                        ray.get(
                            shared_storage.get_info.remote(
                                "num_played_steps")))
                while (ratio < self.config.ratio) and (
                        ray.get(
                            shared_storage.get_info.remote("training_step")) <
                        self.config.training_steps) and not ray.get(
                            shared_storage.get_info.remote("terminate")):
                    time.sleep(0.5)

        self.close_game()

    def play(self, render: bool = False):
        observation = self.env.reset()
        done = False

        replay_data_list = []
        if render:
            self.env.render()

        while not done:
            encoded_state = self.state_enc(observation)
            action = self.actor(encoded_state) + tf.random.normal([2])
            next_observation, reward, done = self.env.step(action)

            if render:
                self.env.render()

            replay_data_list.append(
                ReplayData(observation, action, reward, next_observation,
                           done))

        return replay_data_list

    def close_game(self):
        self.env.close()


class ReplayData:

    def __init__(self, observation, action, reward, next_observation, done):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation
        self.done = done
