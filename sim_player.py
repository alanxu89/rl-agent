import time
import math
import os

import ray
import numpy as np
import tensorflow as tf

from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage
from models import RepresentationNetwork, ActorNetwork
from env import SimulationEnv


@ray.remote
class SimPlayer:

    def __init__(self, config, seed, initial_checkpoint):
        self.config = config
        self.data_files = [
            "/home/alanquantum/Downloads/waymo_motion_data/"
            "uncompressed_scenario_training_training.tfrecord-00000-of-01000"
        ]
        self.env = SimulationEnv(self.data_files)

        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.state_enc = RepresentationNetwork()
        self.actor = ActorNetwork()

        self.state_enc.set_weights(initial_checkpoint["state_encoder_weights"])
        self.actor.set_weights(initial_checkpoint["actor_weights"])

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
                episode_data = self.play(False)
                for frame_data in episode_data:
                    (observation, action, next_observation, reward,
                     done) = frame_data
                    replay_buffer.add.remote(list(observation), action,
                                             list(next_observation), reward,
                                             done, {})
                shared_storage.increment_episodes.remote()

                if self.config.play_delay:
                    time.sleep(self.config.play_delay)

                if self.config.ratio:
                    ratio = ray.get(
                        shared_storage.get_info.remote("training_step")) / max(
                            1,
                            ray.get(
                                shared_storage.get_info.remote(
                                    "num_played_steps")))
                    while (ratio < self.config.ratio) and (ray.get(
                            shared_storage.get_info.remote("training_step")
                    ) < self.config.training_steps) and not ray.get(
                            shared_storage.get_info.remote("terminate")):
                        time.sleep(0.5)
            else:
                episode_data = self.play(False)
                shared_storage.set_info.remote({
                    "episode_length":
                    len(episode_data) - 1,
                    "total_reward":
                    sum([frame_data[3] for frame_data in episode_data]),
                })

        self.close_game()

    def play(self, render: bool = False):
        observation = self.env.reset()
        obs_arrays = convert_observation(observation)
        obs_arrays_expanded = [
            np.expand_dims(arr, axis=0) for arr in obs_arrays
        ]
        done = False

        episode_data = []
        if render:
            self.env.render()

        while not done:
            # inference
            t0 = time.perf_counter()
            encoded_state = self.state_enc(obs_arrays_expanded)
            t1 = time.perf_counter()
            action = self.actor(encoded_state)
            # print(t1 - t0, time.perf_counter() - t1)

            action = tf.squeeze(action) + tf.random.normal([2])

            # simulation
            next_observation, reward, done = self.env.step(action)
            next_obs_arrays = convert_observation(next_observation)

            if render:
                self.env.render()

            episode_data.append(
                (obs_arrays, action, next_obs_arrays, reward, done))

        return episode_data

    def close_game(self):
        self.env.close()


def convert_observation(observation):
    """
    obs: {
            "agent_features": [obs_len, features],
            "social_features": [num_social_agents, obs_len, features],
            "map_features": [[seq_len, features],...]
         }
    """
    agent_feature_arr = convert_agent_feature_to_array(
        observation['agent_features'])
    social_feature_arr = convert_social_feature_to_array(
        observation['social_features'])
    map_feature_arr = convert_map_feature_to_array(observation['map_features'])

    shift = -agent_feature_arr[0, :2]
    dxdy = agent_feature_arr[1, :2] - agent_feature_arr[0, :2]
    angle = -np.arctan2(dxdy[1], dxdy[0])

    agent_feature_arr = coordinate_transform(agent_feature_arr, shift, angle)
    social_feature_arr = coordinate_transform(social_feature_arr, shift, angle)
    map_feature_arr = coordinate_transform(map_feature_arr, shift, angle)

    return (
        agent_feature_arr.astype(np.float32),
        social_feature_arr.astype(np.float32),
        map_feature_arr.astype(np.float32),
    )


def convert_agent_feature_to_array(agent_feature, features_to_keep=9):
    return agent_feature[:, :features_to_keep]


def convert_social_feature_to_array(social_feature,
                                    max_social_agents=96,
                                    features_to_keep=9):
    social_feature = social_feature[:, :, :features_to_keep]
    # pad data
    num_agents = social_feature.shape[0]
    if num_agents >= max_social_agents:
        return social_feature[:max_social_agents]
    else:
        return np.pad(social_feature,
                      ((0, max_social_agents - num_agents), (0, 0), (0, 0)))


def convert_map_feature_to_array(map_feature, max_lanes=64, max_lane_seq=256):
    # pad data
    new_map_feature = []
    for lane_feature in map_feature:
        # print(lane_feature)
        lane_feature = lane_feature['polyline']
        seq_len = lane_feature.shape[0]
        if seq_len >= max_lane_seq:
            lane_feature = lane_feature[:max_lane_seq]
        else:
            lane_feature = np.pad(lane_feature,
                                  ((0, max_lane_seq - seq_len), (0, 0)),
                                  mode='edge')
        new_map_feature.append(lane_feature)
        if len(new_map_feature) >= max_lanes:
            break

    return np.array(new_map_feature)


def coordinate_transform(coords, shift, angle):
    xy = coords[..., :2]
    xy = xy + shift

    x = xy[..., 0]
    y = xy[..., 1]
    x_transform = np.cos(angle) * x - np.sin(angle) * y
    y_transform = np.sin(angle) * x + np.cos(angle) * y
    new_xy = np.stack([x_transform, y_transform], axis=-1)

    coords[..., :2] = new_xy

    return coords
