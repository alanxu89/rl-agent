import time

import tensorflow as tf
import keras
from keras import layers


class RepresentationNetwork(keras.Model):

    def __init__(
        self,
        max_social_agents=32,
        max_lanes=64,
        max_lane_seq=256,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.agent_head = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
        ])

        self.social_head = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
        ])

        self.map_head = keras.Sequential([
            layers.Conv1D(32, 3, activation='relu'),
            layers.AveragePooling1D(),
            layers.Conv1D(64, 3, activation='relu'),
            layers.AveragePooling1D(),
            layers.Conv1D(64, 3, activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
        ])

    def call(self, inputs):
        agent_feature, social_feature, map_feature = inputs

        agent_head_out = self.agent_head(agent_feature)

        social_feat_shape = social_feature.shape
        social_feature = tf.reshape(
            social_feature, [social_feat_shape[0] * social_feat_shape[1], -1])
        social_head_out = self.social_head(social_feature)
        social_head_out = tf.reshape(
            social_head_out, [social_feat_shape[0], social_feat_shape[1], -1])
        social_head_out = tf.reduce_mean(social_head_out, axis=1)

        map_feat_shape = map_feature.shape
        map_feature = tf.reshape(map_feature, [
            map_feat_shape[0] * map_feat_shape[1], map_feat_shape[2],
            map_feat_shape[3]
        ])
        map_head_out = self.map_head(map_feature)
        map_head_out = tf.reshape(map_head_out,
                                  [map_feat_shape[0], map_feat_shape[1], -1])
        map_head_out = tf.reduce_mean(map_head_out, axis=1)

        out = tf.concat([agent_head_out, social_head_out, map_head_out],
                        axis=-1)

        return out


class ValueNetwork(keras.Model):

    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.value = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1),
        ])

    def call(self, x):
        return self.value(x)


class PolicyNetwork(keras.Model):

    def __init__(self, action_space_size):
        super(ValueNetwork, self).__init__()

        self.policy = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(action_space_size),
        ])

    def call(self, x):
        return self.policy(x)


if __name__ == "__main__":
    net = RepresentationNetwork()

    out = net([
        tf.random.normal([64, 6]),
        tf.random.normal([64, 8, 6]),
        tf.random.normal([64, 16, 64, 4])
    ])

    t0 = time.time()
    for _ in range(100):
        out = net([
            tf.random.normal([64, 6]),
            tf.random.normal([64, 8, 6]),
            tf.random.normal([64, 16, 64, 4])
        ])
        print(out.shape)

    print(time.time() - t0)
