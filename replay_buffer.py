import copy
import psutil
import warnings
from typing import Any, Dict, Generator, List, Optional, Union

import ray
import numpy as np


@ray.remote
class ReplayBuffer:
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(self,
                 buffer_size: int = 1000000,
                 action_dim: int = 2,
                 optimize_memory_usage: bool = False,
                 handle_timeout_termination: bool = True):
        self.pos = 0
        self.buffer_size = buffer_size
        self.action_dim = action_dim

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {}
        self.next_observations = {}

        self.actions = np.zeros((self.buffer_size, self.action_dim))
        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size), dtype=np.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if self.next_observations is not None:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        next_obs: Dict[str, np.ndarray],
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            self.observations[key][self.pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            self.next_observations[key][self.pos] = np.array(
                next_obs[key]).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = info.get("TimeLimit.truncated", False)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :return:
        """
        if not self.optimize_memory_usage:
            upper_bound = self.buffer_size if self.full else self.pos
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            return self._get_samples(batch_inds)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (
                np.random.randint(1, self.buffer_size, size=batch_size) +
                self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> Dict:
        obs_ = {
            key: obs[batch_inds, :]
            for key, obs in self.observations.items()
        }
        next_obs_ = {
            key: obs[batch_inds, :]
            for key, obs in self.next_observations.items()
        }

        # Convert to torch tensor
        observations = {key: obs for key, obs in obs_.items()}
        next_observations = {key: obs for key, obs in next_obs_.items()}

        return Dict(
            observations=observations,
            actions=self.actions[batch_inds],
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            rewards=self.rewards[batch_inds],
            dones=self.dones[batch_inds] * (1 - self.timeouts[batch_inds]),
        )


if __name__ == "__main__":
    rpb = ReplayBuffer()
    rpb.add({
        "agent": np.zeros((10, 2)),
        "social": np.zeros((3, 10, 2))
    }, np.zeros(2), {
        "agent": np.zeros((10, 2)),
        "social": np.zeros((3, 10, 2))
    }, 1.0, False, {"TimeLimit.truncated": 1})
