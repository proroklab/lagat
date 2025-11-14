import numpy as np

from pibt.pypibt.pibt import PIBT
from pogema_toolbox.algorithm_config import AlgoBase


class PIBTInferenceConfig(AlgoBase):
    pass


class PIBTInference:
    def __init__(self, config=None, env=None):
        self.env = env
        self.config = config
        self.output_data = None
        self.step = 1
        self.env = env
        self.pibt_expert = None
        if env is not None:
            self.MOVES = np.array(self.env.grid_config.MOVES)

    def reset_states(self, env=None):
        self.step = 1
        self.timed_out = False
        self.pibt_expert = None
        if env is not None:
            self.env = env
            self.MOVES = np.array(self.env.grid_config.MOVES)

    def run_pibt(self):
        obstacles = self.env.grid.get_obstacles(ignore_borders=True)
        starts = self.env.grid.get_agents_xy(ignore_borders=True)
        goals = self.env.grid.get_targets_xy(ignore_borders=True)

        starts = [tuple(s) for s in starts]
        goals = [tuple(g) for g in goals]

        self.pibt_expert = PIBT(
            obstacles == 0, starts, goals, self.env.grid_config.seed
        )
        return self.pibt_expert.run(self.env.grid.config.max_episode_steps + 1)

    def _get_next_move(self, step):
        if step >= len(self.output_data):
            return [0] * self.env.grid_config.num_agents
        diff = self.output_data[step] - self.output_data[step - 1]
        actions = np.argmax(
            np.all(self.MOVES[None, :, :] == diff[:, None, :], axis=-1), axis=-1
        )
        return actions.tolist()

    def act(
        self, observations=None, rewards=None, dones=None, info=None, skip_agents=None
    ):
        if self.output_data is None:
            output_data = self.run_pibt()
            self.output_data = np.array(output_data)
        actions = self._get_next_move(self.step)
        self.step += 1
        return actions
