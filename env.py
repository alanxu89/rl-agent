from abc import ABC, abstractmethod
from typing import List
import random
import math

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
        self.l_r = 1.0
        self.l_f = 1.85
        self.dt = 0.1

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
        """ kinematic bicycle model 

        Args:
            action: [u1, u2]
            where u1 is acceleration and u2 is the front wheel steering
        """
        # get current state
        x, y, v, theta = self.agent_state[:4]

        # extract acceleration and front wheel steering from action input
        u1, u2 = action

        # the slip angle at the center of gravity
        beta = math.atan(math.tan(u2) * self.l_r / (self.l_f + self.l_r))

        x += v * math.cos(theta + beta)
        y += v * math.sin(theta + beta)
        theta += v / self.l_r * math.sin(beta)
        v += u1 * self.dt

        self.agent_state[:4] = np.array([x, y, v, theta])

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
