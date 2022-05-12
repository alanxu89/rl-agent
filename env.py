from abc import ABC, abstractmethod
from typing import List
import random
import math

import numpy as np
from shapely.geometry import Point, LineString, Polygon

from utils import build_polygon, batch_build_polygon


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


class AgentState:

    def __init__(self,
                 x=0.0,
                 y=0.0,
                 z=0.0,
                 l=5.0,
                 w=2.0,
                 h=1.5,
                 heading=0.0,
                 vx=0.0,
                 vy=0.0,
                 valid=False) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.l = l
        self.w = w
        self.h = h
        self.heading = heading
        self.vx = vx
        self.vy = vy
        self.valid = valid

        self.speed = math.sqrt(self.vx**2 + self.vy**2)

        self.lon_polygon_buffer = 0.6
        self.lat_polygon_buffer = 0.2

        self.position = self.__build_agent_position()

        self.polygon = self.__build_agent_polygon()

    def __build_agent_position(self):
        return Point(self.agent_state.x, self.agent_state.y)

    def __build_agent_polygon(self):
        return Polygon(
            build_polygon(self.x, self.y, self.l + self.lon_polygon_buffer,
                          self.w + self.lat_polygon_buffer, self.heading))

    def to_array(self):
        return np.array([
            self.x, self.y, self.z, self.l, self.w, self.h, self.heading,
            self.vx, self.vy, self.valid
        ])


class SimulationEnv(AbstractEnv):

    def __init__(self, data_files: List[str]):
        super().__init__()

        self.data_files = data_files
        self.n_files = len(self.data_files)
        self.cur_data_file = None
        self.scenario = {}

        self.frame_idx = 0
        self.max_frames = 100
        self.obs_len = 10

        self.default_speed_limit = 40.0 / 3.6

        self.agent_state = AgentState()
        self.social_state = None
        # lane speed limit info, etc
        self.map_state = {}
        # light state for agent's current lane, with state and stop_point
        self.light_state = {"state": "GO", "stop_point": np.zeros(2)}

        self.agent_states = []

        self.l_r = 1.0
        self.l_f = 1.85
        self.dt = 0.1
        self.reward_coeff = {
            'dst_reward': 1.0,
            'speed_reward': 0.1,
            'collision_penalty': -1.0,
            'off_lane_penalty': -1.0,
            'violate_traffic_light_penalty': -1.0
        }

    def reset(self):
        self.frame_idx = 0
        self.cur_data_file = self.data_files[random.randint(
            0, self.n_files - 1)]

        self.scenario['agent_features'] = np.random.normal([100, 10])
        agent_init_state = self.scenario['agent_features'][self.frame_idx]
        self.agent_state = AgentState(agent_init_state[0], agent_init_state[1],
                                      agent_init_state[2], agent_init_state[3],
                                      agent_init_state[4], agent_init_state[5],
                                      agent_init_state[6], agent_init_state[7],
                                      agent_init_state[8], agent_init_state[9])
        self.agent_states.append(self.agent_state)
        self.scenario['agent_dst'] = self.scenario['agent_features'][-1, :2]

        self.scenario['social_features'] = np.random.normal([32, 100, 10])
        self.social_state = self.scenario['social_features'][:, self.frame_idx]

        self.scenario['map_features'] = np.random.normal([64, 256, 2])

        self.scenario['light_features'] = np.random.normal([4, 3])

    def step(self, action):
        self.frame_idx += 1

        # update agent
        self.agent_state = self.__apply_vehicle_dynamics(action)

        # update social agents
        self.social_state = self.scenario['social_features'][:, self.frame_idx]

        self.agent_states.append(self.agent_state)

        new_observation = {}
        new_observation['agent_features'] = np.array(
            [s.to_array() for s in self.agent_states[-self.obs_len:]])
        new_observation['social_features'] = self.scenario['social_features'][
            max(0, self.frame_idx - self.obs_len):self.frame_idx]
        new_observation['map_features'] = self.scenario['map_features']
        new_observation['light_features'] = self.scenario['light_features']

        reward = self.__get_reward()

        done = (self.frame_idx >= self.max_frames - 1)

        return new_observation, reward, done

    def random_action(self):
        acc = np.random.normal(scale=2.0)
        steer = np.random.normal(scale=0.1)

        return np.array([acc, steer])

    def __apply_vehicle_dynamics(self, action):
        """ update agent state with kinematic bicycle model 

        Args:
            action: [u1, u2]
            where u1 is acceleration and u2 is the front wheel steering
        """
        x = self.agent_state.x
        y = self.agent_state.y
        theta = self.agent_state.heading
        v = self.agent_state.speed

        # extract acceleration and front wheel steering from action input
        u1, u2 = action

        # the slip angle at the center of gravity
        beta = math.atan(math.tan(u2) * self.l_r / (self.l_f + self.l_r))

        x += v * math.cos(theta + beta)
        y += v * math.sin(theta + beta)
        theta += v / self.l_r * math.sin(beta)
        v += u1 * self.dt
        vx = v * math.cos(theta)
        vy = v * math.sin(theta)

        # update agent state
        new_agent_state = AgentState(x, y, self.agent_state.z,
                                     self.agent_state.l, self.agent_state.w,
                                     self.agent_state.h, theta, vx, vy,
                                     self.agent_state.valid)
        return new_agent_state

    def __get_reward(self):
        reward = 0.0
        for reward_term_name, reward_term_coeff in self.reward_coeff.items():
            reward_func_name = "__get_" + reward_term_name
            reward_func = getattr(self, reward_func_name)
            reward += reward_term_coeff * reward_func()

        return reward

    def __get_dst_reward(self):
        dst = Point(self.scenario['agent_dst'][0],
                    self.scenario['agent_dst'][1])

        return self.agent_state.polygon.intersects(dst)

    def __get_speed_reward(self):
        current_speed_limit = self.map_state.get('speed_limit',
                                                 self.default_speed_limit)
        v = self.agent_state.speed

        v_normalized = v / current_speed_limit

        if v_normalized < 1.0:
            return v_normalized
        else:
            return 1.0 - 10.0 * (v_normalized - 1.0)

    def __get_collision_penalty(self):
        social_state_arr = self.social_state[:, [0, 1, 3, 4, 6]]
        social_polygons = batch_build_polygon(social_state_arr)

        for social_polygon in social_polygons:
            sp = Polygon(social_polygon)
            if self.agent_state.polygon.intersects(sp):
                return 1.0

        return 0

    def __get_off_lane_penalty(self):
        lane_center = self.map_state.get('lane', None)
        road_lines = self.map_state.get('road_line', [])
        road_edges = self.map_state.get('road_edge', [])

        penalty = 0
        if lane_center is not None:
            center_line = LineString(lane_center)
            distance_to_centerline = self.agent_state.position.distance(
                center_line)
            penalty += 0.01 * distance_to_centerline**2

        for road_line in road_lines:
            road_separate_line = LineString(road_line)
            if self.agent_state.polygon.intersects(road_separate_line):
                penalty += 0.1

        for road_edge in road_edges:
            edge_line = LineString(road_edge)
            if self.agent_state.polygon.intersects(edge_line):
                penalty += 0.5

        return penalty

    def __get_violate_traffic_light_penalty(self):
        if self.light_state['state'] == "STOP":
            stop_point = self.light_state['stop_point']

            agent_to_stop_point = np.array([
                stop_point[1] - self.agent_state.y,
                stop_point[0] - self.agent_state.x
            ])
            vec_projection = agent_to_stop_point * np.array([
                math.cos(self.agent_state.heading),
                math.sin(self.agent_state.heading)
            ])

            v = self.agent_state.speed

            if vec_projection < 0:
                # agent is not behind stop point
                return min(abs(vec_projection), 20) * 0.05 * min(v, 30) / 30.0

        return 0
