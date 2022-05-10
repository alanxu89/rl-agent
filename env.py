from abc import ABC, abstractmethod
from typing import List
import random

import numpy as np


class AbstractEnv(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        """
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def random_action(self):
        pass


class SimulationEnv(AbstractEnv):

    def __init__(self, data_files: List[str]):
        super().__init__()

        self.data_files = data_files
        self.n_files = len(self.data_files)
        self.cur_data_file = None
        self.scenario = {}
        self.frame_idx = 0
        self.obs_len = 10
        self.agent_state = None
        self.agent_states = []

    def reset(self):
        self.frame_idx = 0
        self.cur_data_file = self.data_files[random.randint(
            0, self.n_files - 1)]
        self.scenario['agent_features'] = np.random.normal([100, 5])
        self.agent_state = self.scenario['agent_features'][0]
        self.agent_states.append(self.agent_state)
        self.scenario['agent_dst'] = self.scenario['agent_features'][-1, :2]
        self.scenario['social_features'] = np.random.normal([32, 100, 5])
        self.scenario['map_features'] = np.random.normal([64, 256, 2])
        self.scenario['light_features'] = np.random.normal([4, 3])

    def step(self, action):
        # self.agent_state[:2] += 0.1 * action
        self.__apply_vehicle_dynamics(action)
        self.agent_states.append(self.agent_state)

        new_observation = {}
        new_observation['agent_features'] = np.array(self.agent_state[-10:])
        new_observation['social_features'] = self.scenario['social_features'][
            max(0, self.frame_idx - 10):self.frame_idx]
        new_observation['map_features'] = self.scenario['map_features']
        new_observation['light_features'] = self.scenario['light_features']

        reward = self.__get_reward()

        self.frame_idx += 1
        done = (self.frame_idx > 100)

        return new_observation, reward, done

    def __apply_vehicle_dynamics(self, action):
        self.agent_state[:2] += 0.1 * action

    def __get_reward(self):
        return self.__get_dst_reward() + self.__get_collision_penalty(
        ) + self.__get_off_lane_penalty() + self.__get_speed_limit_penalty(
        ) + self.__get_obey_traffic_light_penalty()

    def __get_dst_reward(self):
        return 0

    def __get_collision_penalty(self):
        return 0

    def __get_off_lane_penalty(self):
        return 0

    def __get_speed_limit_penalty(self):
        return 0

    def __get_obey_traffic_light_penalty(self):
        return 0
