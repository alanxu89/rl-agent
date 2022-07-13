from abc import ABC, abstractmethod
from typing import List
import random
import math
import time

import numpy as np
from shapely.geometry import Point, LineString, Polygon

from data_converter import WaymoDataConverter
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
        return Point(self.x, self.y)

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
        self.cur_map_state = {}
        # light state for agent's current lane, with state and stop_point
        self.cur_light_state = {"state": "GO", "stop_point": np.zeros(2)}
        self.lane_light_states = None
        self.agent_hist_states = []

        self.dst = None

        self.start_to_dst = 0.0

        self.done = False

        self.l_r = 1.0
        self.l_f = 1.85
        self.dt = 0.1
        self.reward_coeff = {
            'dst_reward': 1.0,
            'speed_penalty': -0.1,
            'collision_penalty': -1.0,
            'off_lane_penalty': -1.0,
            'violate_traffic_light_penalty': -1.0
        }

        self.data_converter = WaymoDataConverter()
        self.data_converter.read(self.data_files)

    def reset(self):
        self.done = False
        self.frame_idx = self.obs_len
        scenario = self.data_converter.get_a_scenario(idx=0)
        print("sceanrio id {}".format(scenario['scenario_id']))

        self.max_frames = len(scenario['timestamps'])

        agent_track = scenario['tracks_to_predict'][0]
        self.scenario['agent_features'] = scenario['tracks'][agent_track]

        # self.obs_len history steps + current step
        for i in range(self.obs_len + 1):
            state = self.scenario['agent_features'][i]
            self.agent_hist_states.append(
                AgentState(state[0], state[1], state[2], state[3], state[4],
                           state[5], state[6], state[7], state[8], state[9]))
        self.agent_state = self.agent_hist_states[-1]

        self.dst = Point(self.scenario['agent_features'][-1, 0],
                         self.scenario['agent_features'][-1, 1])

        self.start_to_dst = self.agent_state.position.distance(self.dst)

        self.scenario['social_features'] = np.delete(scenario['tracks'],
                                                     agent_track, 0)
        self.social_state = self.scenario['social_features'][:, self.frame_idx]

        self.scenario['map_features'] = {
            'lanes':
            scenario['lanes'],
            'lane_polylines':
            [LineString(lane['polyline']) for lane in scenario['lanes']],
            'road_edges':
            scenario['road_edges'],
            'road_lines':
            scenario['road_lines'],
            'road_lines_type':
            scenario['road_lines_type']
        }

        self.scenario['light_features'] = scenario['dynamics_map_states']

        observation = {}
        observation['agent_features'] = np.array(
            [s.to_array() for s in self.agent_hist_states[-self.obs_len:]])
        observation['social_features'] = self.scenario[
            'social_features'][:,
                               max(0, self.frame_idx - self.obs_len +
                                   1):self.frame_idx + 1]
        observation['map_features'] = self.scenario['map_features']['lanes']
        observation['light_features'] = self.scenario['light_features']

        return observation

    def step(self, action):
        if self.done:
            return None, None, None

        self.frame_idx += 1

        # update agent
        self.agent_state = self.__apply_vehicle_dynamics(action)
        self.agent_hist_states.append(self.agent_state)

        # update social agents
        self.social_state = self.scenario['social_features'][:, self.frame_idx]

        self.lane_light_states = self.scenario['light_features'][
            self.frame_idx]

        self.cur_map_state = self.__find_cur_map_state()

        self.cur_light_state = self.__find_cur_light_state()

        observation = {}
        observation['agent_features'] = np.array(
            [s.to_array() for s in self.agent_hist_states[-self.obs_len:]])
        observation['social_features'] = self.scenario[
            'social_features'][:,
                               max(0, self.frame_idx - self.obs_len +
                                   1):self.frame_idx + 1]
        observation['map_features'] = self.scenario['map_features']['lanes']
        observation['light_features'] = self.scenario['light_features']

        reward = self.__get_reward()

        self.done = self.__dst_reached() or (self.frame_idx
                                             == self.max_frames - 1)

        return observation, reward, self.done

    def render(self):
        # TODO(alanxu): render lanes, road lines, road edges, agent/social
        # positions and light state(color/stop_point)
        pass

    def close(self):
        pass

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

    def __find_cur_map_state(self):
        min_dist_to_lane = 1e7
        min_idx = -1
        idx = 0
        for lane_line in self.scenario['map_features']['lane_polylines']:
            dist_to_lane = self.agent_state.position.distance(lane_line)
            if dist_to_lane < min_dist_to_lane:
                min_dist_to_lane = dist_to_lane
                min_idx = idx
            idx += 1

        if min_idx < 0:
            return {}

        cur_lane = self.scenario['map_features']['lane_polylines'][min_idx]
        cur_lane_spd_lmt = self.scenario['map_features']['lanes'][min_idx][
            'speed_limit']
        if cur_lane_spd_lmt < 1.0:
            cur_lane_spd_lmt = self.default_speed_limit

        return {
            "lane_id": self.scenario['map_features']['lanes'][min_idx]['id'],
            "lane": cur_lane,
            "speed_limit": cur_lane_spd_lmt,
            # TODO(alanxu): add road lines and edges data
            "road_lines": [],
            "road_edges": [],
        }

    def __find_cur_light_state(self):
        if 'lane_id' not in self.cur_map_state:
            # if cur lane is not available
            return {"state": "GO", "stop_point": np.zeros(2)}

        cur_lane_id = self.cur_map_state.get('lane_id')
        cur_light_state = "GO"
        stop_point = np.zeros(2)
        for lane_light_state in self.lane_light_states:
            if cur_lane_id == lane_light_state['lane']:
                light_state = lane_light_state['state']
                if light_state == 1 or light_state == 4 or light_state == 7:
                    cur_light_state = "STOP"
                    stop_point = lane_light_state['stop_point']
                    break
        return {
            "state": cur_light_state,
            "stop_point": stop_point,
        }

    def __get_reward(self):
        reward = 0.0
        for reward_term_name, reward_term_coeff in self.reward_coeff.items():
            reward_func_name = "get_" + reward_term_name
            reward_func = getattr(self, reward_func_name)
            reward += reward_term_coeff * reward_func()

        return reward

    def __dst_reached(self):
        return self.agent_state.polygon.intersects(self.dst)

    def get_dst_reward(self):
        if self.__dst_reached():
            frames_pct_spent = self.frame_idx / (self.max_frames - 1)
            return 1.0 - 0.5 * frames_pct_spent

        if self.done:
            distance_to_dst = self.agent_state.position.distance(self.dst)
            return 0.5 * (1.0 - distance_to_dst / self.start_to_dst)
        else:
            return 0.0

    def get_speed_penalty(self):
        current_speed_limit = self.cur_map_state.get('speed_limit',
                                                     self.default_speed_limit)
        v = self.agent_state.speed

        v_normalized = v / current_speed_limit

        return max(v_normalized - 1.0, 0.0)

    def get_collision_penalty(self):
        # use x, y, l, w, heading columns to build polygon
        social_state_arr = self.social_state[:, [0, 1, 3, 4, 6]]
        social_polygons = batch_build_polygon(social_state_arr)

        for social_polygon in social_polygons:
            sp = Polygon(social_polygon)
            if self.agent_state.polygon.intersects(sp):
                return 1.0

        return 0

    def get_off_lane_penalty(self):
        lane_center = self.cur_map_state.get('lane', None)
        road_lines = self.cur_map_state.get('road_lines', [])
        road_edges = self.cur_map_state.get('road_edges', [])

        penalty = 0
        if lane_center is not None:
            distance_to_centerline = self.agent_state.position.distance(
                lane_center)
            penalty += 0.01 * distance_to_centerline**2

        for road_line in road_lines:
            if self.agent_state.polygon.intersects(road_line):
                penalty += 0.1

        for road_edge in road_edges:
            if self.agent_state.polygon.intersects(road_edge):
                penalty += 0.5

        return penalty

    def get_violate_traffic_light_penalty(self):
        if self.cur_light_state['state'] == "STOP":
            stop_point = self.cur_light_state['stop_point']

            agent_to_stop_point = np.array([
                stop_point[1] - self.agent_state.y,
                stop_point[0] - self.agent_state.x
            ])
            vec_projection = np.dot(
                agent_to_stop_point,
                np.array([
                    math.cos(self.agent_state.heading),
                    math.sin(self.agent_state.heading)
                ]))

            v = self.agent_state.speed

            if vec_projection < 0:
                # agent is not behind stop point
                return min(abs(vec_projection), 20) * 0.05 * min(v, 30) / 30.0

        return 0


if __name__ == "__main__":
    sim_env = SimulationEnv([
        "/home/alanquantum/Downloads/waymo_motion_data/"
        "uncompressed_scenario_training_training.tfrecord-00000-of-01000"
    ])

    obs = sim_env.reset()
    print(obs.keys())
    print(obs.get('agent_features').shape)
    print(obs.get('social_features').shape)
    print(len(obs.get('map_features')))
    print(len(obs.get('light_features')))

    print((obs.get('light_features')))

    # print(obs)

    t0 = time.time()
    for _ in range(100):
        sim_env.reset()
        for i in range(90):
            _, _, done = sim_env.step(np.random.normal(size=(2)))
            if done:
                print(i)
                break
    print(time.time() - t0)
