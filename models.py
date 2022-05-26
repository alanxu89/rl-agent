import time
import copy

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


class CriticNetwork(keras.Model):

    def __init__(self, encoded_state_space=192, action_space=2):
        super(CriticNetwork, self).__init__()

        self.encoded_state_space = encoded_state_space
        self.action_space = action_space

        self.action_head = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
        ])

        self.concat = layers.Concatenate(axis=-1)

        self.dense11 = layers.Dense(64, activation='relu')
        self.dense12 = layers.Dense(64, activation='relu')
        self.dense13 = layers.Dense(1, activation='relu')

        self.dense21 = layers.Dense(64, activation='relu')
        self.dense22 = layers.Dense(64, activation='relu')
        self.dense23 = layers.Dense(1, activation='relu')

        self.__build_model()

    def __build_model(self):
        self.build([(None, self.encoded_state_space),
                    [None, self.action_space]])

    def call(self, inputs):
        """
        inputs: [encoded_state, action]
        """
        state, action = inputs
        action_out = self.action_head(action)
        # print(state.shape)
        # print(action_out.shape)
        state_action = self.concat([state, action_out])
        # print(state_action.shape)

        out = self.dense11(state_action)
        out = self.dense12(out)
        q1 = self.dense13(out)

        out = self.dense21(state_action)
        out = self.dense22(out)
        q2 = self.dense23(out)
        return q1, q2


class ActorNetwork(keras.Model):

    def __init__(self, encoded_state_space=192, action_space=2):
        super(ActorNetwork, self).__init__()

        self.encoded_state_space = encoded_state_space
        self.policy = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(action_space),
        ])

        self.__build_model()

    def __build_model(self):
        self.build((None, self.encoded_state_space))

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

    critic = CriticNetwork()
    print(critic.summary())

    critic_target = CriticNetwork()
    critic_target.set_weights(critic.get_weights())
    print(critic_target.summary())
