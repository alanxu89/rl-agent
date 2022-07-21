import copy
import os
import ray


@ray.remote
class SharedStorage:

    def __init__(self, config, checkpoint):
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)

    def save_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(self.config.results_path, "model_checkpoints")
        self.config.model.save_weights(path)

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError

    def increment_episodes(self, new_episodes=1):
        self.current_checkpoint['num_played_episodes'] += new_episodes

    def get_batch(self):
        return {}
