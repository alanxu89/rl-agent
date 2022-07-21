from abc import ABC, abstractmethod
import os
import time
import random

import numpy as np
import tensorflow as tf

from waymodataset.scenario_pb2 import Scenario


class DataConverter:
    """ convert motion data to sim env data
    """

    @abstractmethod
    def read(self, data_files):
        pass

    @abstractmethod
    def get_a_scenario(self):
        pass


class WaymoDataConverter(DataConverter):

    def __init__(self) -> None:
        super().__init__()

        self.dataset = None
        self.num_records = 0
        self.record_idx = -1
        self.mph_to_mps = 1609.0 / 3600

    def read(self, data_files):
        self.dataset = list(tf.data.TFRecordDataset(data_files))
        self.num_records = len(self.dataset)

    def get_a_scenario(self, idx=None, mode='sequential'):
        """ get scenario by idx or sequential or random
        """
        if idx is not None:
            self.record_idx = idx
        elif mode == "random":
            self.record_idx = random.randint(0, self.num_records - 1)
        else:
            self.record_idx += 1
        raw_record = self.dataset[self.record_idx]
        scenario = self.__parse_record(raw_record.numpy())

        ts = scenario.timestamps_seconds

        tracks_to_predict = []
        for to_predict in scenario.tracks_to_predict:
            tracks_to_predict.append(to_predict.track_index)

        tracks = []
        for track in scenario.tracks:
            track_data = []
            for state in track.states:
                track_data.append([
                    state.center_x, state.center_y, state.center_z,
                    state.length, state.width, state.height, state.heading,
                    state.velocity_x, state.velocity_y, state.valid
                ])
            tracks.append(track_data)
        tracks = np.array(tracks)

        lanes = []
        for map_feature in scenario.map_features:
            if map_feature.HasField('lane'):
                if len(map_feature.lane.polyline) < 2:
                    continue
                polyline = []
                for point in map_feature.lane.polyline:
                    polyline.append([point.x, point.y])
                polyline = np.array(polyline)
                lane_spd_lmt = map_feature.lane.speed_limit_mph * self.mph_to_mps
                lanes.append({
                    'id': map_feature.id,
                    'polyline': polyline,
                    "speed_limit": lane_spd_lmt,
                })

        road_edges = []
        for map_feature in scenario.map_features:
            if map_feature.HasField('road_edge'):
                polyline = []
                for point in map_feature.road_edge.polyline:
                    polyline.append([point.x, point.y])
                polyline = np.array(polyline)
                road_edges.append(polyline)

        road_lines = []
        road_lines_type = []
        for map_feature in scenario.map_features:
            if map_feature.HasField('road_line'):
                polyline = []
                for point in map_feature.road_line.polyline:
                    polyline.append([point.x, point.y])
                polyline = np.array(polyline)
                road_lines.append(polyline)
                road_lines_type.append(map_feature.road_line.type)

        dynamics_map_states = []
        for dms in scenario.dynamic_map_states:
            dynamic_states = []
            for lane_state in dms.lane_states:
                dynamic_state = {}
                dynamic_state['lane'] = lane_state.lane
                dynamic_state['state'] = lane_state.state
                dynamic_state['stop_point'] = np.array(
                    [lane_state.stop_point.x, lane_state.stop_point.y])
                dynamic_states.append(dynamic_state)
            dynamics_map_states.append(dynamic_states)

        scenario_data = {
            'scenario_id': scenario.scenario_id,
            'timestamps': ts,
            'tracks_to_predict': tracks_to_predict,
            'tracks': tracks,
            'lanes': lanes,
            'road_edges': road_edges,
            'road_lines': road_lines,
            'road_lines_type': road_lines_type,
            'dynamics_map_states': dynamics_map_states
        }

        return scenario_data

    def __parse_record(self, raw_record):
        example = Scenario()
        example.ParseFromString(raw_record)
        return example


if __name__ == "__main__":
    waymo = WaymoDataConverter()
    data_dir = "/home/alanquantum/Downloads/waymo_motion_data"
    fname = "uncompressed_scenario_training_training.tfrecord-00001-of-01000"
    waymo.read(os.path.join(data_dir, fname))
    t0 = time.time()
    for _ in range(1):
        s = waymo.get_a_scenario(idx=0)
        print(s['scenario_id'])
    print(time.time() - t0)
    print(s['tracks'].shape)
    tracks = s['tracks']
    # print(tracks[:, :, -1])
    print(np.sum(tracks[:, :, -1]))
    time.sleep(5)
